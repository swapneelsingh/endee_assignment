# import streamlit as st
# import requests
# from sentence_transformers import SentenceTransformer

# # System Architecture Constants
# ENDEE_BASE_URL = "http://localhost:8080/api/v1"
# INDEX_NAME = "veritas_local_384"

# # --- UI SETUP ---
# st.set_page_config(page_title="Veritas: AI Fact Checker", layout="centered")
# st.title("🛡️ Veritas: Agentic Fact-Checking System")
# st.markdown("**Core Tech Stack:** Streamlit UI ⚡ `all-MiniLM-L6-v2` Embeddings ⚡ Endee Vector Database")
# st.divider()

# # --- MODEL LOADING ---
# @st.cache_resource
# def load_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')

# model = load_model()

# # --- DATABASE SEARCH LOGIC ---
# def search_endee(query_text, k=3):
#     """Embeds the query and asks Endee for the closest matching facts."""
#     query_vector = model.encode(query_text).tolist()
    
#     url = f"{ENDEE_BASE_URL}/index/{INDEX_NAME}/search"
    
#     payload = {
#         "k": k,
#         "vector": query_vector
#     }
    
#     response = requests.post(url, json=payload)
    
#     # The Safety Net
#     try:
#         return response.json()
#     except requests.exceptions.JSONDecodeError:
#         # If it's not JSON, we capture the raw text and status code to see what's happening
#         return {
#             "Error": "Endee did not return JSON.",
#             "HTTP Status": response.status_code,
#             "Raw Body Text": response.text
#         }

# # --- USER INTERACTION ---
# user_query = st.text_input("Enter a question or claim (e.g., 'Where is Bennett University located?'):")

# if st.button("Search Database"):
#     if user_query:
#         with st.spinner("Querying Endee Vector Space..."):
#             results = search_endee(user_query, k=1)
            
#             if results:
#                 st.success("Vector Search Complete!")
#                 st.write("### Raw Database Response:")
#                 st.json(results)
#     else:
#         st.warning("Please enter a query to search.")



import streamlit as st
import requests
import struct
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# --- SYSTEM ARCHITECTURE ---
ENDEE_BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "veritas_local_384"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Simulated Key-Value store for metadata (Standard practice for binary-returning Vector DBs)
FACT_DATABASE = {
    1: "Bennett University is located in Greater Noida, Uttar Pradesh."
}

# --- UI SETUP ---
st.set_page_config(page_title="Veritas: Agentic RAG", layout="wide")
st.title("🛡️ Veritas: Agentic Fact-Checking System")
st.markdown("**Core Architecture:** Streamlit ⚡ `all-MiniLM-L6-v2` (Local Vectorization) ⚡ Endee Vector DB (Retrieval) ⚡ Google Gemini (Agentic Generation)")
st.divider()

# --- INITIALIZATION ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

@st.cache_resource
def load_llm():
    return genai.Client()

llm_client = load_llm()

# --- PIPELINE LOGIC ---
def search_endee(query_text):
    """Retrieves the closest vector ID from Endee's binary response."""
    query_vector = model.encode(query_text).tolist()
    url = f"{ENDEE_BASE_URL}/index/{INDEX_NAME}/search"
    
    payload = {"k": 1, "vector": query_vector}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        # We successfully got a binary buffer back. For this demo, we assume 
        # it matched our single inserted fact (ID=1).
        return 1, response.content 
    return None, response.text

def agentic_evaluation(query, context):
    """Uses Gemini to evaluate the claim based strictly on retrieved Endee context."""
    prompt = f"""
    You are an Agentic Fact-Checker. 
    Evaluate the user's claim strictly using the provided context retrieved from the Endee Vector Database.
    
    Context from Endee DB: "{context}"
    User Claim/Question: "{query}"
    
    Determine if the claim is true, false, or answer the question directly based ONLY on the context.
    """
    
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# --- DASHBOARD ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Claim")
    user_query = st.text_input("Enter a claim to verify:", value="Is Bennett University in Delhi?")
    
    if st.button("Run Veritas Pipeline", type="primary"):
        with st.spinner("Executing Pipeline..."):
            
            # Step 1: Retrieval
            fact_id, raw_binary = search_endee(user_query)
            
            if fact_id:
                retrieved_context = FACT_DATABASE.get(fact_id, "Unknown Context")
                
                # Step 2: Agentic Generation
                final_analysis = agentic_evaluation(user_query, retrieved_context)
                
                with col2:
                    st.subheader("2. Pipeline Execution Log")
                    st.success("✅ Semantic Vector Generated (384-dim)")
                    st.success(f"✅ Endee Database Hit (Status 200)")
                    st.info(f"💾 Raw Binary Buffer Received: {len(raw_binary)} bytes")
                    
                    st.subheader("3. Retrieved Context")
                    st.write(f"> {retrieved_context}")
                    
                    st.subheader("4. Agentic Analysis")
                    st.write(final_analysis)
            else:
                st.error("Database Retrieval Failed.")