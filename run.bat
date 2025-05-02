@echo off
echo Starting MLflow...
start /b mlflow ui --port 5000
timeout /t 2 /nobreak >nul

echo Starting FastAPI...
start /b uvicorn main:app --reload --host 127.0.0.1 --port 8000
timeout /t 2 /nobreak >nul

echo Starting Streamlit...
start /b streamlit run ui.py
pause