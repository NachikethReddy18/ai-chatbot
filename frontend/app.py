import os
import requests
import streamlit as st
import pathlib

# ---------------- CONFIG ----------------
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/api/v1/chat"
)

st.set_page_config(
    page_title="AI Customer Support",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD CSS ----------------
css_path = pathlib.Path(__file__).parent / "styles.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="chat-container">
        <div class="app-title">🤖 AI Customer Support</div>
        <div class="app-subtitle">
            Instant help for orders, payments, and technical issues
        </div>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-bubble user'>{msg['text']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bubble bot'>{msg['reply']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class='confidence-wrapper'>
                <div class='confidence-bar' style='width:{msg['confidence']*100}%;'></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.markdown("<div class='status'>🟢 Online • Replies instantly</div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Type your message here..."
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # User message
    st.session_state.messages.append({
        "role": "user",
        "text": user_input
    })

    try:
        res = requests.post(
            API_URL,
            json={"message": user_input},
            timeout=5
        ).json()

        st.session_state.messages.append({
            "role": "bot",
            "reply": res["reply"],
            "confidence": res["confidence"]
        })

        if res.get("escalate"):
            st.warning("⚠ This issue has been escalated to a human agent.")

    except Exception:
        st.error("Backend is unavailable. Please try again later.")

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
