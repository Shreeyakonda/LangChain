from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    # Load and split documents
    print("📘 Loading HR Policy document...")
    loader = TextLoader("hr_policy.txt", encoding='utf-8')
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} document(s)")

    # Split documents into smaller chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")

    # Create embeddings
    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    print("💾 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save for later use
    vectorstore.save_local("faiss_index")
    print("🎉 Vector store created and saved successfully!")

except FileNotFoundError:
    print("❌ Error: 'hr_policy.txt' not found. Please check the file path.")
except Exception as e:
    print(f"⚠️ An error occurred: {type(e).__name__}: {str(e)}")
