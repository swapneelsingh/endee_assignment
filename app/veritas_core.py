import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

ENDEE_BASE_URL = "http://localhost:8080/api/v1"
# NEW INDEX NAME: Bypasses the old 768-dim index completely
INDEX_NAME = "veritas_local_384" 
EMBEDDING_DIMENSION = 384  

print("Loading local AI model... (this takes a few seconds)")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Turns text into a 384-dimensional vector locally."""
    return model.encode(text).tolist()

def setup_endee_index():
    """Tells Endee to create a new index."""
    print(f"Creating index '{INDEX_NAME}' in Endee...")
    payload = {
        "index_name": INDEX_NAME,
        "dim": EMBEDDING_DIMENSION,
        "space_type": "cosine"
    }
    response = requests.post(f"{ENDEE_BASE_URL}/index/create", json=payload)
    
    if response.status_code == 200:
        print("--> Success: Index created.")
    elif response.status_code == 409:
        print("--> Success: Index already exists. Ready to use.")
    else:
        print(f"--> Create Index HTTP Status: {response.status_code}")
        print(f"--> Create Index Raw Body: {response.text}")

def insert_fact_into_endee(fact_id, text):
    """Embeds a fact and stores it in Endee."""
    vector = get_embedding(text)
    url = f"{ENDEE_BASE_URL}/index/{INDEX_NAME}/vector/insert"
    
    # CHANGED: Flattened the JSON since the endpoint is singular ("vector/insert")
    payload = {
        "id": fact_id,
        "vector": vector,
        "metadata": {"text": text}
    }
    
    response = requests.post(url, json=payload)
    print(f"--> Insert Fact HTTP Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"--> Insert Fact Raw Body: {response.text}")

# if __name__ == "__main__":
#     print("\n1. Setting up Endee Index (384-dim)...")
#     setup_endee_index()

#     print("\n2. Inserting a verified fact...")
#     test_fact = "Bennett University is located in Greater Noida, Uttar Pradesh."
#     insert_fact_into_endee(fact_id=1, text=test_fact)
    
#     print("\nPipeline Test Complete.")

if __name__ == "__main__":
    print("\n1. Setting up Endee Index (384-dim)...")
    setup_endee_index()

    print("\n2. Populating Knowledge Base...")
    # A mix of facts, including some MERN stack and tech trivia!
    knowledge_base = [
        "Bennett University is located in Greater Noida, Uttar Pradesh.",
        "The MERN stack consists of MongoDB, Express.js, React, and Node.js.",
        "Docker uses OS-level virtualization to deliver software in packages called containers.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Retrieval-Augmented Generation (RAG) grounds Large Language Models on external knowledge bases."
    ]

    for index, fact in enumerate(knowledge_base):
        # We use index + 1 so our IDs start at 1, 2, 3...
        insert_fact_into_endee(fact_id=index + 1, text=fact)
        print(f"Inserted Fact {index + 1}...")
    
    print("\nPipeline Test & Seeding Complete.")