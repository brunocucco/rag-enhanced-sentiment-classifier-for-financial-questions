import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re, feedparser
from html import unescape

# Available models
MODEL_OPTIONS = {
    "finetuned-128-model": "bcco/finetuned-128-model",
    "finetuned-256-model": "bcco/finetuned-256-model"
}

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def clean_html(text):
    text = re.sub(r'<a.*?>.*?</a>', '', text)
    text = re.sub(r'<.*?>', '', text)
    return unescape(text.strip())

def fetch_news(query, max_results=50):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    cleaned = []
    for entry in feed.entries[:max_results]:
        title = entry.title
        summary = clean_html(entry.get("summary", ""))
        cleaned.append(f"{title}. {summary}".strip())
    return cleaned

def build_faiss_index(news):
    embeddings = embedder.encode(news, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_similar(news, query, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_vec, top_k)
    return [(news[i], distances[0][j]) for j, i in enumerate(indices[0])]

# --- Streamlit UI ---
st.title("📈 RAG-Enhanced Sentiment Analyzer")

# Dropdown to select model
selected_model_name = st.selectbox("Choose a fine-tuned model:", list(MODEL_OPTIONS.keys()))

# Load selected model/tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline_model = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipeline_model

sentiment_pipeline = load_model_and_tokenizer(MODEL_OPTIONS[selected_model_name])
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# User input
query = st.text_input("Ask about a stock or financial topic:", "How is Microsoft stock doing?")

if st.button("Classify"):
    # Fetch and embed news
    with st.spinner("🔎 Retrieving news and running classification..."):
        news = fetch_news(query, max_results=50)
        faiss_index, _ = build_faiss_index(news)
        top_news = retrieve_similar(news, query, top_k=5)
        context = " ".join([snippet for snippet, _ in top_news])
        full_input = f"<NEWS>: {context} <QUERY>: {query}"
        pred = sentiment_pipeline(full_input)[0]
        label_id = int(pred["label"].split("_")[-1])
        label = label_map[label_id]
        score = round(pred["score"], 3)

    # Output
    st.markdown(f"### Classified: **{label}**")
    st.markdown(f"**Confidence:** {score}")
    st.markdown(f"**Model Used:** `{selected_model_name}`")

    st.markdown("### Top News Snippets Used:")
    for i, (snippet, dist) in enumerate(top_news, 1):
        st.write(f"{i}. {snippet}  \n_Distance: {round(dist, 2)}_")

    with st.expander("Full Model Input"):
        st.code(full_input[:1000] + "...", language="text")
