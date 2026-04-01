from dotenv import load_dotenv
load_dotenv()
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(result_type="markdown", api_key=os.environ.get("LLAMA_CLOUD_API_KEY"))
reader = SimpleDirectoryReader(input_dir="data", file_extractor={".pdf": parser})
try:
    docs = reader.load_data()
    print("Docs loaded:", len(docs))
    if docs:
        print("Sample content length:", len(docs[0].text))
        print("First 100 chars:", docs[0].text[:100])
except Exception as e:
    print("Error:", e)
