import streamlit as st
from dotenv import load_dotenv
from lib import RagClient
import tempfile
import os
import logging
import hashlib

load_dotenv(override=True)

def get_file_hash(file_content):
    """Generate a hash for the file content to use as a unique identifier."""
    return hashlib.md5(file_content).hexdigest()

def display_sources(sources):
    with st.expander("View Sources"):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"""
            **Source {i}**
            - Page: {doc.metadata.get('page', 'N/A')}
            - Content: {doc.page_content[:200]}...
            """)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Bot", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "client" not in st.session_state:
    try:
        st.session_state.client = RagClient()
        st.session_state.client.init_retriever()
        st.session_state.client.setup_chain()
    except Exception as e:
        st.error(f"Failed to initialize RAG client: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        st.stop()

st.sidebar.title("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", 
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.sidebar.status("Processing documents...") as status:
        for uploaded_file in uploaded_files:
            # Generate a unique identifier for the file
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            # Only process if we haven't seen this file before
            if file_hash not in st.session_state.processed_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(file_content)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        st.sidebar.info(f"Processing {uploaded_file.name}...")
                        response = st.session_state.client.add_documents(tmp_file_path)
                        if "Successfully stored" in response:
                            st.session_state.processed_files.add(file_hash)
                            st.sidebar.success(f"{uploaded_file.name}: {response}")
                        else:
                            st.sidebar.warning(f"{uploaded_file.name}: {response}")
                    except Exception as e:
                        st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
                    finally:
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            logger.error(f"Error removing temporary file: {str(e)}")
                
                except Exception as e:
                    st.sidebar.error(f"Error handling {uploaded_file.name}: {str(e)}")
                    logger.error(f"File handling error: {str(e)}")
            else:
                st.sidebar.info(f"Skipped {uploaded_file.name} (already processed)")
        
        status.update(label="Document processing complete!", state="complete")
        
# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            display_sources(message["sources"])

if prompt := st.chat_input("Ask your question here"):
    with st.chat_message("user"):
        st.write(prompt)
    
    # adding user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.client.rag_chain.invoke(prompt)
            answer = response["answer"]
            sources = response.get("source_documents", [])
            
            st.write(answer)
            if sources:
                display_sources(sources)
            
            # Add assistant response to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

with st.sidebar:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
