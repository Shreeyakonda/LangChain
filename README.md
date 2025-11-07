# 🤖 HR Policy Chatbot using LangChain, Ollama, FAISS, and Streamlit

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot that answers questions about an organization's HR policy.  
It uses **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Ollama's Llama 3.2 model** to retrieve relevant context and generate accurate responses.  
A **Streamlit interface** provides a simple and interactive web-based chat experience on your local host.

---

## 📚 Features

✅ Answers questions based on HR policies or any custom text document  
✅ Uses RAG (Retrieval-Augmented Generation) pipeline for grounded answers  
✅ FAISS vector database for fast semantic search  
✅ HuggingFace sentence transformer embeddings  
✅ Llama 3.2 model running locally via Ollama  
✅ Streamlit web interface — chat with your HR policy instantly  

---

## 🏗️ Project Structure

LangChain-1

├── hr_policy.txt # HR Policy document (knowledge base)

├── vector_store.py # Creates FAISS vector store from hr_policy.txt

├── rag_demo.py # Tests RAG pipeline in the terminal

├── app.py # Streamlit web app

├── faiss_index/ # Generated vector database (after running vector_store.py)

├── requirements.txt # Python dependencies

├── curr_date_tool.py # LangChain tool demo (date example)

├── prompt_template.py # Prompt demo using LangChain templates

├── demo.py # Simple Ollama LLM example

└── README.md # This file

---

## ⚙️ Prerequisites

### 🐍 Python
Ensure Python 3.10+ is installed.

### 🧰 Required Libraries
Install all dependencies:

```bash
pip install -r requirements.txt
```

### 🦙 Ollama Setup

Install Ollama from "https://ollama.ai/download"
and pull the model:
```
ollama pull llama3.2
```
Then start the Ollama server:
```
ollama serve
```
-------
## 🚀 Step-by-Step Setup
1️⃣ Create a Virtual Environment
```
python -m venv langchain_env
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
pip install streamlit
```
3️⃣ Build the Vector Store
Run this once to create embeddings and the FAISS index:
```
python vector_store.py
```
You should see:
```
🎉 Vector store created and saved successfully!
```
This generates a folder named faiss_index/.

4️⃣ Run the RAG Chatbot (CLI)
To test directly from the terminal:
```
python rag_demo.py
```
You’ll see output like:
```
🧠 Question: What is the company's leave policy?
💬 Answer: The company provides 12 days of casual leave, 10 days of sick leave...
```
5️⃣ Launch the Streamlit Web App

Now run your chatbot locally:
```
streamlit run app.py
```
## 🧠 Technologies Used
| Component                  | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| **LangChain**              | Framework for building LLM pipelines                   |
| **FAISS**                  | Facebook AI Similarity Search for vector storage       |
| **HuggingFace Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` for embedding |
| **Ollama**                 | Local LLM runner (Llama 3.2 model)                     |
| **Streamlit**              | Frontend web interface for chatbot                     |
| **Python**                 | Core programming language                              |

### 💡 Future Improvements

- 🗨️ Add chat history (memory) for multi-turn conversation

-  📂 Allow users to upload their own HR or policy documents

- 🎨 Add chat-style UI with avatars and scrolling history

- ☁️ Deploy on cloud (Streamlit Cloud, HuggingFace Spaces, or local Docker)
