# 🛡️ Veritas: Agentic Fact-Checking System

> A high-performance Retrieval-Augmented Generation (RAG) pipeline designed for real-time claim verification and fake news detection.


## 🚀 Overview & Use Case
Veritas is an Agentic RAG system built to combat misinformation by instantly verifying user claims against a trusted, vectorized knowledge base. Rather than relying solely on an LLM's internal (and potentially hallucinated) memory, Veritas forces the AI to ground its analysis strictly on verified facts retrieved dynamically from the **Endee Vector Database**.

## 🧠 System Architecture Pipeline

The system is designed for zero-latency embeddings and high-speed semantic retrieval:

```mermaid
graph TD
    A[User Inputs Claim] -->|Streamlit UI| B(Local Embedding Engine)
    B -->|all-MiniLM-L6-v2| C{Endee Vector Database}
    C -->|Cosine Similarity Search| D[Retrieve Binary ID Buffer]
    D -->|Map to Verified Fact| E(Google Gemini 2.5 Flash)
    E -->|Agentic Evaluation| F[Final Verification Output]
    
    style C fill:#007acc,stroke:#333,stroke-width:2px,color:#fff

