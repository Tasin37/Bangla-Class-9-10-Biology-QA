import streamlit as st
import faiss
import pickle
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bangla Biology RAG QA",
    layout="wide"
)

st.title("🧬 Bangla Biology Question Answering (RAG) (Class 9-10)")
st.markdown(
    "Retrieval-Augmented Generation for **factual Bangla educational QA**"
)

# -----------------------------
# Load resources (cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/LaBSE")

@st.cache_resource
def load_index():
    index = faiss.read_index("bangla_bio.index")
    with open("bangla_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource
def load_llm():
    model_name = "bigscience/bloomz-1b1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

embedder = load_embedder()
index, chunks_meta = load_index()
tokenizer, model, device = load_llm()

# -----------------------------
# Helper functions (same as notebook)
# -----------------------------
def split_sentences(text):
    return re.split(r'[।!?]\s*', text)

def retrieve_top_sentences(question, k_chunks=3, k_sent=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    _, I = index.search(q_emb, k_chunks)

    sentences = []
    for idx in I[0]:
        for s in split_sentences(chunks_meta[idx]["text"]):
            if len(s.strip()) > 20:
                sentences.append(s.strip())

    if not sentences:
        return []

    sent_emb = embedder.encode(sentences, convert_to_numpy=True)
    faiss.normalize_L2(sent_emb)

    sims = sent_emb @ q_emb.T
    top_ids = np.argsort(sims.squeeze())[::-1][:k_sent]

    return [sentences[i] for i in top_ids]

def build_rag_prompt(question, sentences):
    context = "\n".join([f"- {s}" for s in sentences])

    prompt = f"""
আপনি একজন জীববিজ্ঞান শিক্ষক।

নিয়ম:
- সর্বোচ্চ ১–২টি বাক্য লিখবেন
- শুধুমাত্র দেওয়া তথ্য ব্যবহার করবেন
- অতিরিক্ত ব্যাখ্যা করবেন না
- তথ্য না থাকলে লিখবেন: "উত্তর দেওয়া সম্ভব নয়"

তথ্য:
{context}

প্রশ্ন: {question}

উত্তর (১–২ বাক্য):
""".strip()

    return prompt

def generate_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.2,
            length_penalty=0.8,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean prompt echo
    if "উত্তর" in text:
        text = text.split("উত্তর", 1)[-1]

    return text.strip()

# -----------------------------
# UI
# -----------------------------
question = st.text_input(
    "📝 প্রশ্ন লিখুন (Bangla):",
    placeholder="যেমন: রক্ত কী ধরনের কলা?"
)

col1, col2 = st.columns(2)
with col1:
    k_chunks = st.slider("🔍 Top Chunks", 1, 5, 3)
with col2:
    k_sent = st.slider("📌 Top Sentences", 1, 5, 3)

if st.button("উত্তর দিন"):
    if not question.strip():
        st.warning("অনুগ্রহ করে একটি প্রশ্ন লিখুন")
    else:
        with st.spinner("তথ্য অনুসন্ধান ও উত্তর তৈরি হচ্ছে..."):
            retrieved = retrieve_top_sentences(
                question,
                k_chunks=k_chunks,
                k_sent=k_sent
            )
            prompt = build_rag_prompt(question, retrieved)
            answer = generate_answer(prompt)

        st.subheader("✅ উত্তর")
        st.success(answer)

        st.subheader("📚 ব্যবহৃত তথ্য (Evidence)")
        if retrieved:
            for i, s in enumerate(retrieved, 1):
                st.markdown(f"**{i}.** {s}")
        else:
            st.info("কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি")
