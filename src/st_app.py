import hashlib
import logging
import os
import tempfile

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from lib import RagClient

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()


def display_sources(sources):
    with st.expander("View Sources"):
        for i, doc in enumerate(sources, 1):
            st.markdown(
                f"""**Source {i}**
            - Page: {doc.metadata.get('page', 'N/A')}
            - Content: {doc.page_content[:200]}...
            """
            )


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}   # the hash is used for storing metadata
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "client" not in st.session_state:
        try:
            st.session_state.client = RagClient()
            st.session_state.client.init_retriever()
            st.session_state.client.setup_chain()
        except Exception as e:
            st.error(f"Failed to initialize RAG client: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            st.stop()


def handle_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                display_sources(message["sources"])

    if prompt := st.chat_input("Ask your question here"):
        with st.chat_message("user"):
            st.write(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.client.rag_chain.invoke(prompt)
                st.write(response["answer"])

                if "source_documents" in response:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["source_documents"],
                        }
                    )


def process_documents(uploaded_files):
    """Process only new documents once during upload"""
    processed_results = []
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)

        if file_hash in st.session_state.processed_files:
            st.sidebar.info(f"Skipped {uploaded_file.name} (already processed)")
            continue

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"temp_{file_hash}.pdf")
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            st.sidebar.info(f"Processing {uploaded_file.name}...")
            response = st.session_state.client.add_documents(temp_path)
            
            if "Successfully stored" in response:
                # Store metadata about processed file
                st.session_state.processed_files[file_hash] = {
                    'name': uploaded_file.name,
                    'timestamp': datetime.now().isoformat()
                }
                st.sidebar.success(f"{uploaded_file.name}: {response}")
                processed_results.append({
                    'name': uploaded_file.name,
                    'status': 'success',
                    'hash': file_hash
                })
        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            processed_results.append({
                'name': uploaded_file.name,
                'status': 'error',
                'error': str(e)
            })
        finally:
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {str(e)}")
    
    st.session_state.documents_loaded = True
    return processed_results


def main():
    st.set_page_config(page_title="RAG Bot", layout="wide")
    init_session_state()

    st.sidebar.title("Document Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files and not st.session_state.documents_loaded:
        with st.sidebar.status("Processing documents...") as status:
            results = process_documents(uploaded_files)
            logger.info(results)

    handle_chat()


if __name__ == "__main__":
    main()
