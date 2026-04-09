import streamlit as st
import os
from dotenv import load_dotenv

# Load our core logic from the rag_chatbot script!
from rag_chatbot import load_documents, split_text, build_vector_database, create_rag_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Q&A Chatbot")
st.markdown("Ask anything based on the PDFs stored in the `pdfs/` directory.")

# 1. Environment Check
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found. Please add it to your `.env` file.")
    st.stop()

# 2. Setup the RAG Pipeline
# We use @st.cache_resource so we don't have to reload and re-embed documents
# every single time you type a question. It stores the chain in memory.
@st.cache_resource(show_spinner="Booting up Vector Database...")
def initialize_rag():
    documents = load_documents("pdfs")
    if not documents:
        return None
    chunks = split_text(documents)
    # The Chroma DB persists locally, so it uses cached embeddings if nothing changed
    vector_store = build_vector_database(chunks, persist_directory="./chroma_db")
    rag_chain = create_rag_chain(vector_store)
    return rag_chain

rag_chain = initialize_rag()

if not rag_chain:
    st.warning("No PDFs found in the 'pdfs/' directory. Please add some to begin your Q&A!")
    st.stop()

# 3. Streamlit Chat Interface 
# Keep track of message history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Immediately render user message to screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and render the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer")
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Render sources in an expander nicely hidden
                with st.expander("📚 View Sources Used"):
                    for i, doc in enumerate(response.get("context", [])):
                        file_name = doc.metadata.get('file_name', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        chunk_idx = doc.metadata.get('chunk_index', 'Unknown')
                        words = doc.metadata.get('word_count', 'Unknown')
                        
                        st.write(f"**[{i+1}] File:** `{file_name}` | **Page:** {page} | **Chunk Index:** {chunk_idx} | **Words:** {words}")
                        st.caption(f'"{doc.page_content[:200]}..."')
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
