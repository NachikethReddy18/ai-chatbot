# 🤖 AI Customer Support Chatbot

An end-to-end AI-powered customer support chatbot that classifies user intent using a fine-tuned Transformer model and responds with realistic, business-style answers.

## Features
- Intent classification using DistilBERT
- FastAPI backend
- Rule-based response engine (real-world style)
- Modern, animated Streamlit frontend
- Confidence-based escalation logic

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch
- FastAPI
- Streamlit

## Run Locally

### Backend & Frontend
```bash
uvicorn src.api.app:app --reload  (backend)
streamlit run frontend/app.py     (frontend)


