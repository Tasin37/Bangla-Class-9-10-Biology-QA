# Bangla-Class-9-10-Biology-QA
# 🇧🇩 Bangla Educational Question Answering using RAG

This project enhances factual accuracy in Bangla educational question answering by combining **Retrieval-Augmented Generation (RAG)** with **Large Language Models (LLMs)**.

## 🚀 Features
- Bangla biology QA
- FAISS-based dense retrieval
- LaBSE multilingual embeddings
- BLOOMZ language model
- Streamlit web application
- Reduced hallucination compared to baseline LLM

## 🧠 Architecture
1. User question
2. FAISS retrieves relevant Bangla textbook chunks
3. Retrieved sentences passed as context
4. LLM generates concise answer (1–2 sentences)

## 📊 Results
| Metric | Baseline | RAG |
|------|---------|-----|
| Token F1 | 0.0381 | **0.2277** |
| Semantic Similarity | 0.2891 | **0.5902** |
| Hallucinations | 107 | **1** |

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
