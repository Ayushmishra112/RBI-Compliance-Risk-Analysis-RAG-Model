import chromadb
db = chromadb.PersistentClient(path="chroma_db")
col = db.get_collection("rbi_compliance")
data = col.get() # fetch all
print("Total vectors:", len(data['ids']))
if data['ids']:
    print("Metadata sample:", data['metadatas'][0])
    # check how many have 2017 vs 2026
    y_2017 = sum(1 for m in data['metadatas'] if m.get('year') == 2017)
    y_2026 = sum(1 for m in data['metadatas'] if m.get('year') == 2026)
    print(f"2017 count: {y_2017}, 2026 count: {y_2026}")
