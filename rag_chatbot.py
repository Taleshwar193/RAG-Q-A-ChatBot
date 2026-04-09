import os
import glob
from dotenv import load_dotenv

# --- Step 1: Load Environment Variables ---
# We use python-dotenv to load the .env file if it exists.
# This keeps secrets like our OpenAI key out of the source code.
load_dotenv()

# Verify that our API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_documents(pdf_directory="pdfs"):
    """
    Step 2: Load multiple PDFs into our project from a directory using PyPDFLoader.
    """
    print(f"Loading PDFs from directory: {pdf_directory} ...")
    
    # We grab paths for all .pdf files inside the specified directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in '{pdf_directory}'. Please add some PDFs to test the RAG pipeline.")
        return []

    documents = []
    # Loop over every PDF we found
    for pdf_path in pdf_files:
        print(f" - Loading: {pdf_path}")
        # Initialize the PyPDFLoader for the specific PDF file
        loader = PyPDFLoader(pdf_path)
        # Load the content and append to our overall list of documents
        docs = loader.load()
        
        # --- Adding Document-Level Metadata ---
        for doc in docs:
            # os is already imported at the top of the file
            doc.metadata["file_name"] = os.path.basename(pdf_path)
            doc.metadata["document_format"] = "PDF"
            
        documents.extend(docs)
        
    print(f"Loaded {len(documents)} document pages total.")
    return documents

def split_text(documents, embeddings=None):
    """
    Step 3: Split the full text into smaller semantic chunks using SemanticChunker.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    
    print("Splitting text into chunks using Semantic Chunking...")
    
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
        
    text_splitter = SemanticChunker(embeddings)
    
    # Perform the splitting
    chunks = text_splitter.split_documents(documents)
    
    # --- Adding Chunk-Level Metadata ---
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["word_count"] = len(chunk.page_content.split())
        
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def build_vector_database(chunks, persist_directory="./chroma_db"):
    """
    Step 4 & 5: Utilize OpenAI embeddings and store them in ChromaDB.
    ChromaDB allows us to perform semantic vector search later.
    """
    print("Initializing OpenAI Embeddings...")
    # Initialize embedding model. Make sure you have credit in your OpenAI account!
    embeddings = OpenAIEmbeddings()
    
    print("Building / loading Chroma DB vector store...")
    # Create the vector store from our chunks and embedding model.
    # Specifying persist_directory tells Chroma to save data to disk.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def create_rag_chain(vector_store):
    """
    Step 6 & 7: Set up the OpenAI LLM, retriever, and the RAG response pipeline.
    """
    print("Setting up LLM and Retriever...")
    
    # 6. Initialize our Chat Model (Using OpenAI)
    # Using 'gpt-4o-mini' or 'gpt-3.5-turbo' is usually cost-effective for generic RAG tasks
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,max_tokens=2000)
    
    # 7. Convert our vector store into a Retriever object.
    # The retriever defines HOW we search (e.g. similarity) and how many chunks to return (k=3)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Define the System Prompt telling the LLM exactly how it should answer:
    # "You are an assistant for question-answering tasks..."
    system_prompt = (
        "You are an AI assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context: {context}"
    )

    # Let's create a template structure for our prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Here we bind our LLM with our prompt to handle combinations of retrieved documents
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Final step: Tie the retriever and the Document QA chain together into a single workflow.
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def interactive_loop(rag_chain):
    """
    Enables users to have a back and forth QA session through the terminal.
    """
    print("\n" + "="*50)
    print("RAG System Ready! Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("\nAsk a question: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting chatbot.")
            break
            
        if not user_input.strip():
            continue
            
        try:
            print("Thinking...")
            # We pass our string to the 'input' key defined in our prompt
            response = rag_chain.invoke({"input": user_input})
            
            # The structure of the RAG chain output holds 'answer' for the final text
            print("\nAnswer:", response.get("answer"))
            print("\nSources Used:")
            for i, doc in enumerate(response.get("context", [])):
                file_name = doc.metadata.get('file_name', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                chunk_idx = doc.metadata.get('chunk_index', 'Unknown')
                words = doc.metadata.get('word_count', 'Unknown')
                print(f"  [{i+1}] {file_name} (Page {page}) - Chunk Index: {chunk_idx} [{words} words]")
                
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    # Execute the entire workflow
    
    # 1/2. Load your raw documents
    documents = load_documents("pdfs")
    
    if not documents:
        print("Cannot proceed without documents. Quitting.")
        return
        
    # 3. Create intelligent chunks
    chunks = split_text(documents)
    
    # 4/5. Embed chunks and save/load them into a Database
    vector_store = build_vector_database(chunks)
    
    # 6/7. Tie everything together into a Retrieval Augmented Generation pipeline
    rag_chain = create_rag_chain(vector_store)
    
    # 8. Start our chatbot!
    interactive_loop(rag_chain)

if __name__ == "__main__":
    main()
