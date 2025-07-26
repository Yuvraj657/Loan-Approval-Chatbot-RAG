
#  RAG Q&A Chatbot – Loan Approval Insights using Gemini

This is a Retrieval-Augmented Generation (RAG) chatbot that intelligently answers questions about loan approval, based on a real-world dataset. It retrieves relevant records from the dataset and uses **Gemini (Google Generative AI)** to generate human-like answers.

---

##  Features

-  **Semantic Search** with FAISS + Sentence Transformers
-  **Context-Aware Answering** via Gemini API (`gemini-2.0-flash`)
-  Based on [Loan Approval Dataset from Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
- Built with Python & Streamlit 

---

##  Project Structure

```

rag_loan_chatbot/
├── app.py                 # Streamlit app
├── rag.py           # RAG logic (retrievalGemini)
├── requirements.txt       # Required Python packages
├── Training Dataset.csv   # Input dataset from Kaggle
└── README.md              # Project documentation

````

---

##  Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/rag_loan_chatbot.git
cd rag_loan_chatbot
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your Gemini API key**

```bash
export GOOGLE_API_KEY="your-gemini-api-key"  # Mac/Linux
# or
set GOOGLE_API_KEY="your-gemini-api-key"     # Windows
```

4. **Download the dataset**

* Download `Training Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
* Place it in the project root folder.

---

##  Run the App

```bash
streamlit run app.py
```

---

##  Example Questions You Can Ask

* "Does credit history affect loan approval?"
* "How does self-employment impact the chances?"
* "What if the applicant is not married and has a low income?"

---

##  How It Works

1. Uses **Sentence Transformers** to embed the dataset rows.
2. Stores them in a **FAISS** index for fast similarity search.
3. On user query:

   * Finds top relevant entries.
   * Sends them as context to **Gemini (via `gemini-2.0-flash`)**.
   * Returns an intelligent, context-aware response.

---

##  Dependencies

* `streamlit`
* `pandas`
* `faiss-cpu`
* `sentence-transformers`
* `google-generativeai`

Install them with:

```bash
pip install -r requirements.txt
```


---

Made by Yuvraj Singh
AI-DS BE IVth Year


