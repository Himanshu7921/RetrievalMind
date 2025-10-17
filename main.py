from src.data_ingestion import PDFDocumentIngestor
from src.embeddings_manager import EmbeddingManager
from src.vector_store_manager import VectorStore
from src.rag_retriver import Retrieval

# Step 1: Load and chunk the PDF document
pdf_file_path = "data/pdf/Company_Policy_Document.pdf"  # Path to the company policy PDF

# Initialize the PDF loader
pdf_ingestor = PDFDocumentIngestor(
    file_path=pdf_file_path,
    encoding='utf-8',
    loader_type='mu'  # 'mu' uses PyMuPDFLoader (faster), 'std' uses PyPDFLoader (standard)
)

# Load the PDF into LangChain document objects
pdf_loader = pdf_ingestor.load_document()
document_chunks = pdf_loader.load()  # List of document chunks (LangChain documents)

# Extract the text content from each chunk for embedding
chunk_texts = [chunk.page_content for chunk in document_chunks]

# Step 2: Generate embeddings for each chunk
# Initialize the embedding manager with default model
embedding_manager = EmbeddingManager()  # Uses 'all-miniLM-L6-v2' model by default

# Generate embeddings for all chunks
chunk_embeddings = embedding_manager.generate_embeddings(chunk_texts)

# Step 3: Store document chunks and embeddings in a vector store
vector_store_name = "company_policy_collection"  # Name of the ChromaDB collection
vector_store_directory = "data/vector_store"     # Directory to persist the vector store

# Initialize the vector store
vector_store = VectorStore(
    collection_name=vector_store_name,
    persist_directory=vector_store_directory,
    document_type="PDF"
)

# Add all chunks and their embeddings to the vector store
vector_store.add_document(
    documents=document_chunks,
    embeddings=chunk_embeddings
)

# Step 4: Initialize retrieval pipeline
retrieval_pipeline = Retrieval(vector_store=vector_store, embedding_manager=embedding_manager)

# Define a query to search in the document store
query_text = "What is the termination policy of the company?"

# Retrieve the top matching document
retrieved_results = retrieval_pipeline.retrieve(
    query=query_text,
    top_k=1,           # Return only the top document
    score_threshold=0  # Include all matches above this similarity score
)

# Extract the content of the top matching document
if retrieved_results:
    top_document_content = retrieved_results[0]['content']
    print("Retrieved Document Content:\n", top_document_content)
else:
    print("No relevant documents found for the query.")