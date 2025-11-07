from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load embeddings and FAISS index
print("📂 Loading FAISS vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("✅ Vector store loaded successfully!")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Create RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:
""")

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Query example
question = "What is the company's leave policy?"
print(f"\n🧠 Question: {question}\n")

answer = rag_chain.invoke(question)
print(f"💬 Answer: {answer}")
