import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams


class RagClient:
    def __init__(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        # self.llm = ChatOpenAI(model_name="gpt-4o")

        self.collection_name = os.getenv(
            "QDRANT_COLLECTION_NAME", "compliance_prototype"
        )

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
            )

        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=70, length_function=len
        )

        try:
            self.qdrant_client.get_collections()
            print("Successfully connected to Qdrant server.")
        except UnexpectedResponse as e:
            print(f"Error connecting to Qdrant server: {e}")
            print(f"Qdrant URL: {qdrant_url}")
            print(
                "Please check your QDRANT_URL and QDRANT_API_KEY environment variables."
            )
        self.create_collection()

    def create_collection(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(
                collection.name == self.collection_name for collection in collections
            )

            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"Created new collection: {self.collection_name}")

            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        except UnexpectedResponse as e:
            print(f"Error accessing Qdrant collection: {e}")
            raise

    # TODO: Please rename the method, circular callbacks are bad
    def add_documents(self, file: str = "../docs/guide to peft.pdf"):
        # loader = PyPDFLoader("../docs/LoRA.pdf")
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        chunks = self.text_splitter.split_documents(pages)
        self.vector_store.add_documents(chunks)

        return f"Successfully stored {len(chunks)} document chunks in database."

    def init_retriever(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Changed to MMR for diversity
            search_kwargs={
                "k": 5,
                "fetch_k": 20,  # Fetch more candidates initially
                "lambda_mult": 0.7,  # Controls diversity (0.0-1.0). Higher = more similarity focused
            },
        )

    def setup_chain(self):
        PROMPT_TEMPLATE = """
        You are an expert tutor who will be a teaching assistant. 
        Answer questions based on the given context. Be direct and specific.
        
        Guidelines:
        - Always base your answers on the provided context
        - Synthesize information from different sources
        - For follow-up questions, refer to previous chat history when relevant
        - If information isn't in the context, say "I cannot find this information in the provided documents"
        - Include specific references to the source material when possible
        - Each piece of information should cite its source page number

        Context:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Please provide a comprehensive answer that combines information from different sources:"""

        self.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            },
            chain_type="stuff",  # Use 'stuff' method for combining documents
            verbose=True
        )
