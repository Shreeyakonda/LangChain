import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Page Config ---
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 HR Policy RAG Chatbot")
st.write("Ask questions about your company's HR policy")

# --- Load FAISS index ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Initialize LLM ---
llm = ChatOllama(model="llama3.2", temperature=0.3)

# --- Prompt Template ---
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:
""")

# --- Format retrieved docs ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Build RAG chain ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Chat Interface ---
st.markdown("### 💬 Ask a question about the HR Policy")

user_input = st.text_input("Your question:", placeholder="e.g., What is the company's leave policy?")
if st.button("Ask") or user_input:
    with st.spinner("Thinking... 🤔"):
        answer = rag_chain.invoke(user_input)
        st.markdown("#### 🧠 Answer:")
        st.success(answer)
