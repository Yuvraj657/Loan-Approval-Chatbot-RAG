import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load full dataset once
df_full = pd.read_csv("Training Dataset.csv")

# Load data as documents
def load_data():
    docs = []
    for _, row in df_full.iterrows():
        doc = f"Gender: {row['Gender']}, Married: {row['Married']}, Education: {row['Education']}, " \
              f"Self_Employed: {row['Self_Employed']}, ApplicantIncome: {row['ApplicantIncome']}, " \
              f"LoanAmount: {row['LoanAmount']}, Credit_History: {row['Credit_History']}, " \
              f"Loan_Status: {row['Loan_Status']}"
        docs.append(doc)
    return docs

# Embed and index documents using Sentence Transformers + FAISS
def embed_and_index(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(docs)
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(doc_embeddings))
    doc_id_map = {i: doc for i, doc in enumerate(docs)}
    return model, index, doc_id_map

# Retrieve top-k most similar documents
def retrieve(query, model, index, doc_id_map, k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [doc_id_map[i] for i in indices[0]]

# Optional: generate statistics based on question content
def get_stats_if_relevant(query):
    stats = ""

    if "married" in query.lower():
        married_stats = df_full.groupby('Married')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
        stats += "Approval Rate by Marital Status:\n" + str(married_stats) + "\n"

    if "credit" in query.lower():
        credit_stats = df_full.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
        stats += "Approval Rate by Credit History:\n" + str(credit_stats) + "\n"

    if "income" in query.lower():
        avg_income = df_full.groupby('Loan_Status')['ApplicantIncome'].mean()
        stats += "Average Applicant Income by Loan Status:\n" + str(avg_income) + "\n"

    if "self-employed" in query.lower():
        se_stats = df_full.groupby('Self_Employed')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0)
        stats += "Approval Rate by Self-Employment Status:\n" + str(se_stats) + "\n"

    return stats or "No additional statistical insights found."

# Gemini generation with injected context and real stats
def generate_answer_with_gemini(query, context_docs, extra_stats=None):
    # Secure key loading
    genai.configure(api_key="AIzaSyD62sGUTlM58z0y_2RDKFg6hAemwZrOyeM")
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')

    context = "\n".join(context_docs)
    stats = extra_stats or ""

    prompt = f"""
You are an intelligent assistant answering questions using a real-world loan approval dataset.

First, read the data context and statistics provided. Then answer the question clearly and accurately.

--- Dataset Context ---
{context}

--- Statistical Insights ---
{stats}

--- Question ---
{query}

--- Answer ---
"""

    response = model_gemini.generate_content(prompt)
    return response.text.strip()

# Main RAG chatbot function
def rag_chatbot(query, model, index, doc_id_map):
    docs = retrieve(query, model, index, doc_id_map)
    stats = get_stats_if_relevant(query)
    return generate_answer_with_gemini(query, docs, stats)
