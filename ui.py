import streamlit as st
import requests
from PIL import Image

# FastAPIのエンドポイントURL
FASTAPI_URL = "http://localhost:8000/predict"

st.title("画像分類アプリ")

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードした画像", use_container_width=True)
    
    # FastAPIに画像を送信
    with st.spinner("分類中..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()            
            st.success(result)
        else:
            st.error("エラーが発生しました。")