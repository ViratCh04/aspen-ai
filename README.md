# Aspen AI

Aspen AI is a Retrieval-Augmented Generation (RAG) application designed to serve as an expert teaching assistant. It leverages advanced language models to provide accurate and contextually relevant answers based on uploaded documents. This project is ideal for educational purposes, showcasing how to build a well-performing RAG client.

## Features
- **Document Upload**: Upload multiple PDF documents to create a knowledge base.
- **Conversational Interface**: Engage in a dialogue with the AI assistant to ask questions about the content.
- **Contextual Answers**: The assistant provides answers based on the uploaded documents, citing specific sources and page numbers.
- **Source Display**: View the exact sources used to generate the answers.

## Getting Started 🚀

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/docs/) for dependency management.
- OpenAI API Key: Required for language model access.
- Qdrant API URL and Key: For the vector database.

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/ViratCh04/aspen-ai.git
    ```
2. Install dependencies:
    ```
    poetry install
    ```
3. Activate the virtual environment:
    ```
    poetry shell(to switch to virtual environment)
    ```
4. Set up environment variables:
    
    Create a .env file in the root directory with the following content:
    ```
    OPENAI_API_KEY=your_openai_api_key
    QDRANT_URL=your_qdrant_url
    QDRANT_API_KEY=your_qdrant_api_key
    QDRANT_COLLECTION_NAME=your_collection_name (optional)
    ```

### Usage
<b>Streamlit Interface</b>
To run the Streamlit interface:

```
cd src/
streamlit run st_app.py
```
<b>Chainlit Interface</b>
Alternatively, you can run the Chainlit interface:

```
cd src/
chainlit run cl_app.py -w
```

## How It Works
### Architecture Overview
- **Document Loading and Chunking**: PDF documents are loaded and split into manageable chunks using a text splitter.
- **Vector Store**: Chunks are embedded and stored in a Qdrant vector database.
- **Retriever**: Implements Maximum Marginal Relevance (MMR) to fetch diverse and relevant document chunks.
- **Conversational Chain**: Utilizes a language model (e.g., GPT-3.5-turbo) to generate answers based on the retrieved context.
### Key Components
- `lib.py`: Contains the RagClient class that handles document processing, vector store interactions, retriever initialization, and chain setup.
- `st_app.py`: Streamlit application providing a user interface for document upload and chat interaction.
- `cl_app.py`: Chainlit application alternative for chat interaction.

### Detailed Explanation
- **Loading Documents**: Uses PyPDFLoader to read PDF files.
- **Text Splitting**: Utilizes RecursiveCharacterTextSplitter to divide text into chunks for processing.
- **Embedding and Storing**: Chunks are embedded using OpenAI embeddings and stored in the Qdrant vector store.


<b>Retriever Configuration (lib.py)</b>

```python
def init_retriever(self):
    self.retriever = self.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )
```

- **MMR (Maximum Marginal Relevance)**: Retrieves results that balance relevance and diversity.
- **Parameters**:
    - `k`: Number of documents to return.
    - `fetch_k`: Number of candidate documents to consider.
    - `lambda_mult`: Controls trade-off between similarity and diversity (0.0 - 1.0).


<b>Conversational Chain Setup (lib.py)</b>

```python
def setup_chain(self):
    PROMPT_TEMPLATE = """
    your prompt goes here!
    """

    self.rag_chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=self.retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        }
    )
```

- **Prompt Template**: Guides the assistant to provide context-aware answers.
- **Chain Configuration**: Manages how the assistant interacts with the user and retrieves information.

<b>Streamlit App (st_app.py)</b>

```python
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
        with st.sidebar.spinner("Processing documents..."):
            process_documents(uploaded_files)

    handle_chat()
```
- **Session State Management**: Keeps track of messages and processed files to avoid reprocessing.
- **File Upload Handling**: Processes new documents only upon upload.
- **Chat Interface**: Provides an interactive chat where users can ask questions.


## Contributing
Contributions are welcome! Fork, commit, push and PR!