
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import shutil

# ---- Page Config ----
st.set_page_config(
    page_title="☁️ Cloud Sentiment Analyzer",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Social Media Sentiment Analyzer")
st.markdown("**Cloud-based NLP pipeline using VADER + DistilBERT**")
st.divider()

# ---- Load Models ----
@st.cache_resource
def load_models():
    vader = SentimentIntensityAnalyzer()
    bert = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    return vader, bert

with st.spinner("Loading AI models... (first run takes ~30 seconds)"):
    vader_analyzer, bert_pipeline = load_models()

# ---- Single Tweet Analysis ----
st.header("🔍 Analyze a Tweet or Text")
user_input = st.text_area(
    "Enter your text here:",
    placeholder="Type a tweet, review, or any social media post...",
    height=100
)

if st.button("Analyze Sentiment 🚀", type="primary"):
    if user_input.strip():
        col1, col2 = st.columns(2)

        # VADER
        scores = vader_analyzer.polarity_scores(user_input)
        compound = scores["compound"]
        vader_label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")

        with col1:
            st.subheader("🔵 VADER")
            color = "green" if vader_label == "Positive" else ("red" if vader_label == "Negative" else "gray")
            st.markdown(f"**Sentiment:** :{color}[{vader_label}]")
            st.metric("Compound Score", f"{compound:.4f}")
            fig = go.Figure(go.Bar(
                x=["Positive", "Negative", "Neutral"],
                y=[scores["pos"], scores["neg"], scores["neu"]],
                marker_color=["#2ecc71", "#e74c3c", "#95a5a6"]
            ))
            fig.update_layout(title="Score Breakdown", height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # BERT
        bert_result = bert_pipeline(user_input[:512], truncation=True)[0]
        bert_label = "Positive" if bert_result["label"] == "POSITIVE" else "Negative"
        bert_conf = bert_result["score"]

        with col2:
            st.subheader("🟣 DistilBERT")
            color2 = "green" if bert_label == "Positive" else "red"
            st.markdown(f"**Sentiment:** :{color2}[{bert_label}]")
            st.metric("Confidence", f"{bert_conf:.2%}")
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bert_conf * 100,
                title={"text": "Confidence %"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#9b59b6"}}
            ))
            fig2.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Agreement
        if vader_label == bert_label:
            st.success(f"✅ Both models agree: **{bert_label}** sentiment!")
        else:
            st.warning(f"⚠️ Models disagree — VADER says **{vader_label}**, BERT says **{bert_label}**")
    else:
        st.error("Please enter some text!")

# ---- Bulk CSV Analysis ----
st.divider()
st.header("📂 Bulk CSV Analysis")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must have a 'text' column!")
    else:
        sample = df.head(100)  # Limit for speed
        with st.spinner("Analyzing sentiments..."):
            sample["vader_sentiment"] = sample["text"].apply(
                lambda t: "Positive" if vader_analyzer.polarity_scores(str(t))["compound"] >= 0.05
                else ("Negative" if vader_analyzer.polarity_scores(str(t))["compound"] <= -0.05 else "Neutral")
            )
        st.dataframe(sample[["text", "vader_sentiment"]].head(20))
        fig3 = px.pie(sample, names="vader_sentiment",
                      color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
                      title="Sentiment Distribution")
        st.plotly_chart(fig3, use_container_width=True)

        csv_out = sample.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv_out, "sentiment_output.csv", "text/csv")

# ---- Footer ----
st.divider()
st.caption("Built with ❤️ using HuggingFace Transformers, VADER, and Streamlit | Cloud Deployment via Streamlit Cloud")
