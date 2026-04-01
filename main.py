import os
import warnings
warnings.filterwarnings("ignore", message=".*parsing_instruction is deprecated.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")
import nest_asyncio
try:
    nest_asyncio.apply()
except ValueError:
    pass
import streamlit as st
import chromadb
from llama_index.llms.groq import Groq

# Configuration Flag for Sidebar
SHOW_SIDEBAR = False
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

class SafeSubQuestionQueryEngine(SubQuestionQueryEngine):
    def _query_subq(self, sub_q, color=None):
        if sub_q.tool_name not in self._query_engines:
            valid_tools = list(self._query_engines.keys())
            for vt in valid_tools:
                if vt in sub_q.tool_name:
                    sub_q.tool_name = vt
                    break
            if sub_q.tool_name not in self._query_engines:
                sub_q.tool_name = valid_tools[0]
        return super()._query_subq(sub_q, color=color)
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.question_gen import LLMQuestionGenerator
try:
    from llama_parse import LlamaParse
except RuntimeError:
    LlamaParse = None
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="RBI Compliance & Risk Analysis RAG Model", layout="wide", page_icon="🏦")



DIR_DATA = "data"
DIR_CHROMA = "chroma_db"
os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_CHROMA, exist_ok=True)

groq_key = os.environ.get("GROQ_API_KEY")
llama_cloud_key = os.environ.get("LLAMA_CLOUD_API_KEY")

if groq_key:
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile", 
        api_key=groq_key,
        system_prompt=(
            "You are a Senior Fintech Product Analyst specializing in Indian Banking Regulations.\n"
            "When answering compare rules across the requested years using this EXACT markdown format:\n"
            "**Historical Baseline:** [Summarize the older rule]\n\n"
            "**New Mandate:** [Summarize the newer or proposed rule]\n\n"
            "**Product Action Items:** [Provide 2-3 specific technical or UI changes required]\n"
        )
    )

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ==========================================
# 2. METADATA EXTRACTORS & INGESTION
# ==========================================
def get_meta(file_path):
    import re
    fname = os.path.basename(file_path)
    metadata = {"source": fname}
    
    match = re.search(r'(201[4-9]|202[0-9])', fname)
    if match:
        year = int(match.group(1))
        metadata.update({"year": year, "status": "Draft" if year >= 2026 else "Active"})
    else:
        metadata.update({"year": 2017, "status": "Active"}) # Fallback if no year found
    return metadata

def load_and_index_documents():
    """Reads PDFs from data/, parses them with LlamaParse, and stores in ChromaDB."""
    if not os.listdir(DIR_DATA):
        return False, "No documents found in data/ directory."
        
    if not llama_cloud_key:
        return False, "LLAMA_CLOUD_API_KEY is not set. Cannot run LlamaParse."

    parser = LlamaParse(result_type="markdown", api_key=llama_cloud_key)
    file_extractor = {".pdf": parser}

    try:
        reader = SimpleDirectoryReader(
            input_dir=DIR_DATA,
            file_extractor=file_extractor,
            file_metadata=get_meta
        )
        docs = reader.load_data()
        if not docs:
            return False, "LlamaParse returned 0 documents. (DNS Block?)"
            
        parser_md = MarkdownNodeParser()
        nodes = parser_md.get_nodes_from_documents(docs)
        
    except Exception as e:
        return False, f"LlamaParse Network Error: {str(e)}"

    # Initialize single ChromaDB persistent client
    db = chromadb.PersistentClient(path=DIR_CHROMA)
    chroma_collection = db.get_or_create_collection("rbi_compliance")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index directly from the structurally chunked nodes
    VectorStoreIndex(nodes, storage_context=storage_context)
        
    return True, "Successfully parsed, structurally chunked, and indexed to ChromaDB!"

def get_indexed_years():
    """Reads ChromaDB to find all unique years currently indexed natively."""
    try:
        db = chromadb.PersistentClient(path=DIR_CHROMA)
        col = db.get_collection("rbi_compliance")
        data = col.get()
        if not data or not data.get('metadatas'): return []
        return sorted(list(set(m.get("year") for m in data['metadatas'] if m and "year" in m)))
    except Exception:
        return []

# ==========================================
# 3. AGENTIC REASONING ENGINE SETUP
# ==========================================
def get_query_engine():
    """Configures the SubQuestionQueryEngine targeting the grouped ChromaDB Index."""
    db = chromadb.PersistentClient(path=DIR_CHROMA)
    
    try:
        col = db.get_collection("rbi_compliance")
    except Exception as e:
        return None  # Collections might not exist yet
        
    vector_store = ChromaVectorStore(chroma_collection=col)
    
    # In LlamaIndex, recreating index from vector store without making a new one:
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    data = col.get()
    unique_years = sorted(list(set(m.get("year") for m in data['metadatas'] if "year" in m)))
    if not unique_years:
        unique_years = [2017, 2026] # Fallback

    tools = []
    for year in unique_years:
        tools.append(
            QueryEngineTool(
                query_engine=index.as_query_engine(
                    similarity_top_k=3,
                    filters=MetadataFilters(filters=[ExactMatchFilter(key="year", value=year)])
                ),
                metadata=ToolMetadata(name=f"rbi_{year}", description=f"RBI Guidelines strictly for the year {year}. You MUST query this tool INDIVIDUALLY.")
            )
        )

    query_engine = SafeSubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
        use_async=False
    )
    
    return query_engine

# ==========================================
# 4. STREAMLIT UI
# ==========================================

# Sleek Minimalist Aesthetic CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Minimalist Header */
    h1 {
        color: #0f172a !important;
        font-weight: 600;
        letter-spacing: -0.5px;
        padding-bottom: 5px;
    }
    
    /* Clean Chat Bubbles */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        padding: 1.2rem;
        transition: all 0.2s ease;
    }
    
    /* Flat Minimalist Expander Cards */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: border-color 0.2s ease;
        margin-top: 10px;
    }
    [data-testid="stExpander"]:hover {
        border-color: #94a3b8;
    }
    
    /* Subtle Buttons */
    .stButton>button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton>button:active {
        transform: scale(0.97) !important;
    }
    
    /* Clean Input Box */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: none !important;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stChatInput"]:focus-within {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06) !important;
    }
    /* Terminate Streamlit's Native Blue Outline entirely */
    [data-testid="stChatInput"] *, [data-testid="stChatInput"] *:focus, [data-testid="stChatInput"] *:active {
        outline: none !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("RBI Compliance & Risk Analysis RAG Model")
st.markdown("<p style='color: #64748b; font-weight: 400; font-size: 1.15rem; margin-top: -15px;'>Empowered by Groq Llama-3.3-70B • LlamaParse API</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
if SHOW_SIDEBAR:
    with st.sidebar:
        st.header("⚙️ Active Knowledge Base")
        
        # Display dynamically indexed years from ChromaDB
        indexed_years = get_indexed_years()
        if indexed_years:
            st.success(f"**Index Active For Years:** {', '.join(map(str, indexed_years))}")
        else:
            st.warning("**Indexed Years:** None")
            
        st.divider()
        st.header("Document Management")
        
        if not groq_key:
            st.error("GROQ_API_KEY is not set in `.env`")
        if not llama_cloud_key:
            st.error("LLAMA_CLOUD_API_KEY is not set in `.env`")
            
        uploaded_files = st.file_uploader(
            "Upload RBI Circulars (PDF)", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Extract & Index via LlamaParse"):
            if not LlamaParse:
                st.error("Document extraction is disabled on this server because it runs Python 3.14. Native database queries are fully functional though!")
            else:
                with st.spinner("Parsing PDFs into Markdown via LlamaCloud and indexing..."):
                    if uploaded_files:
                        for uf in uploaded_files:
                            with open(os.path.join(DIR_DATA, uf.name), "wb") as f:
                                f.write(uf.read())
                    
                    success, msg = load_and_index_documents()
                    if success:
                        st.success(msg)
                    else:
                        st.warning(msg)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Chunks"):
                for source in message["sources"]:
                    st.markdown(f"**Source:** {source['metadata'].get('source', 'Unknown')} | **Year:** {source['metadata'].get('year', 'N/A')} | **Priority:** {source['metadata'].get('priority', 'N/A')}")
                    st.text(source['text'])
                    st.divider()

if prompt := st.chat_input("Ask a regulatory compliance query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not groq_key:
            st.error("Cannot process query: GROQ_API_KEY is missing.")
            st.stop()
            
        engine = get_query_engine()
        if not engine:
            st.error("Index not found. Please upload documents and click 'Extract & Index' first.")
            st.stop()
            
        # Basic Conversational Interceptor for Non-RAG Queries
        friendly_greetings = ["hi", "hello", "hey", "help", "who are you?", "what can you do?", "hi there", "sup"]
        if prompt.strip().lower() in friendly_greetings:
            st.success("👋 **Hello!** I am your focused **RBI Compliance & Risk AI**.\n\nI don't engage in generic small talk because my brain is 100% locked strictly to retrieving and comparing Reserve Bank of India documents! Please ask me a specific regulatory question, such as: *'How did the prepaid card guidelines change between 2017 and 2026?'*")
            st.stop()
            
        with st.spinner("Analyzing..."):
            response = engine.query(prompt)
            
            # Handle strict RAG Empty Responses gracefully
            if not response.response or str(response.response).strip() == "Empty Response":
                st.warning("⚠️ **No Regulatory Context Found.** The underlying AI searched the ChromaDB vectors but could not find any explicit RBI mandates matching your query. Try rephrasing with specific banking terms!")
                st.stop()
                
            st.markdown(response.response)
            
            sources_data = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources_data.append({
                        "text": node.node.text,
                        "metadata": node.node.metadata
                    })
                
            if sources_data:
                with st.expander("📚 View Agentic Reasoning & Data Sources"):
                    for source in sources_data:
                        meta = source['metadata']
                        # Identify if node is a raw PDF text chunk vs an AI sub-question calculation
                        if "source" in meta and "year" in meta:
                            st.markdown(f"📄 **Raw PDF Chunk** | **Source:** {meta.get('source')} | **Year:** {meta.get('year')}")
                        else:
                            st.markdown("🤖 **Agentic Sub-Query Logic**")
                            
                        st.text(source['text'])
                        st.divider()
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response.response,
                "sources": sources_data
            })
