import streamlit as st
from transformers import pipeline

st.title("日本語文章の感情分析アプリ")

# モデルの読み込み（初回に少し時間がかかる）
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment")

model = load_model()

# ユーザー入力
text = st.text_area("感情を知りたい文章を入力してください", height=150)

# 分析＆出力
if st.button("感情を分析する") and text:
    with st.spinner("分析中..."):
        result = model(text)
        label = result[0]['label']
        score = result[0]['score']
        st.markdown(f"### 感情：**{label}**（信頼度：{score:.2f}）")
