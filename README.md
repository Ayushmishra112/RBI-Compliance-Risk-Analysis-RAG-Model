# RBI Compliance and Risk Analysis RAG Model

An Agentic Retrieval-Augmented Generation (RAG) system built to automate delta analysis across various Reserve Bank of India (RBI) circulars spanning multiple years (2014-2026). The application provides comparative regulatory insights, compliance gap identification, and actionable product recommendations for fintech operators.

## Architecture and Tech Stack

The system is built on a "Local-First/Free-Stack" architecture prioritizing privacy, speed, and cost-efficiency:

- Large Language Model: Groq (Llama-3.3-70B-Versatile) for high-speed, high-quality reasoning.
- Embeddings: Local HuggingFace embeddings (BAAI/bge-small-en-v1.5) running completely offline.
- Parsing: LlamaParse API specifically configured to ingest and structuralize complex RBI regulatory tables into clean Markdown.
- Structural Chunking: MarkdownNodeParser for context-aware extraction, preventing arbitrary token splitting across related paragraphs.
- Vector Database: ChromaDB configured for persistent local storage.
- Orchestration: LlamaIndex Agentic SubQuestionQueryEngine.
- Interface: Streamlit with a minimalist, flat CSS design system.

## Key Capabilities

1. Dynamic Year Tracking: Automatically parses the timeline of any uploaded RBI PDF using Regex, dynamically building isolated AI routing tools for every distinct year present in the database.
2. Synthetic Sub-Query Generation: The agent automatically breaks complex comparative questions down into independent timeline searches before synthesizing a final response.
3. Strict Output Formatting: The LLM is strictly prompted to return comparative analysis broken down into:
   - Historical Baseline
   - New Mandate
   - Product Action Items
4. Source Attribution Transparency: Exposes the exact agentic reasoning steps and raw PDF citations used to process the response.

## Local Execution

To run the application on your own hardware:

1. Clone this repository.
2. Install the dependencies via `pip install -r requirements.txt`.
3. Create a `.env` file in the root directory and populate it with your API keys:
   GROQ_API_KEY="your-groq-key"
   LLAMA_CLOUD_API_KEY="your-llamacloud-key"
4. Execute `python -m streamlit run main.py`.
5. Upload your target RBI PDF circulars and click "Extract and Index" to build the local ChromaDB network.

## Deployment Notes

This application is designed to be fully compatible with Streamlit Community Cloud. When deploying, ensure that your `chroma_db` directory is pushed to your remote repository so the runtime is instantly initialized with your pre-computed vectors, avoiding redundant LlamaParse API calls on every server reboot.
