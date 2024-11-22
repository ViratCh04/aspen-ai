import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

#load_dotenv(override=True)

class RagClient:
    def __init__(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        #self.llm = ChatOpenAI(model_name="gpt-4o")

        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "compliance_prototype")

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
            )

        self.qdrant_client = QdrantClient(
            url=qdrant_url, api_key=qdrant_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=40,
            length_function=len
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

    def add_documents(self):
        # Testing out the llama paper
        loader = PyPDFLoader("../docs/2302.13971v1 llama.pdf")
        pages = loader.load_and_split()
        chunks = self.text_splitter.split_documents(pages)
        # Storing doc chunks in vector store
        self.vector_store.add_documents(chunks)

    def init_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        #retrieved_docs = retriever.invoke("What is this paper about?")
        #print(len(retrieved_docs))
        #print(retrieved_docs)


    def setup_chain(self):
        #TODO: Create custom prompt, this one is severely limited to only generating three sentences responses at max and can be customised much more to generate better responses specific to our use case
        prompt = hub.pull("rlm/rag-prompt")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
