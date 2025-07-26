from rag import load_data, embed_and_index, rag_chatbot
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Loan Approval Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body, .main {
            background-color: #f0f4f8;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .answer-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-size: 16px;
            color: #333333;
            margin-top: 20px;
        }
        .stButton button {
            background-color: #2a9d8f;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            margin-top: 10px;
        }
        .stButton button:hover {
            background-color: #21867a;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: gray;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #264653;'> Loan Approval Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 17px;'>Ask intelligent questions about loan approval trends</p>", unsafe_allow_html=True)

# --- Input ---
with st.form("qa_form"):
    query = st.text_input("🔍 Type your question:", placeholder="e.g. What factors increase loan approval chances?")
    submitted = st.form_submit_button("Get Answer")

# --- Execution ---
if submitted and query:
    with st.spinner("Analyzing your question with AI and data..."):
        docs = load_data()
        model, index, doc_id_map = embed_and_index(docs)
        answer = rag_chatbot(query, model, index, doc_id_map)

    # --- Answer Display ---
    st.markdown("###  Answer:")
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Built with 💙 using Streamlit + Gemini | Retrieval-Augmented Generation</div>", unsafe_allow_html=True)
