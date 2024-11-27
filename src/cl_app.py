import os
import chainlit as cl
import logging
import hashlib
from dotenv import load_dotenv
from lib import RagClient
import tempfile

load_dotenv(override=True)

def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

class ChainlitRagApp:
    def __init__(self):
        self.client = None
        self.processed_files = set()
        self.setup_rag_client()

    def setup_rag_client(self):
        try:
            self.client = RagClient()
            self.client.init_retriever()
            self.client.setup_chain()
        except Exception as e:
            logging.error(f"Failed to initialize RAG client: {str(e)}")
            raise

    async def process_file(self, file):
        # Generate a unique identifier for the file
        with open(file.path, "rb") as f:
            file_content = f.read()
        file_hash = get_file_hash(file_content)
        
        # Only process if we haven't seen this file before
        if file_hash not in self.processed_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name
                try:
                    response = self.client.add_documents(tmp_file_path)
                    self.processed_files.add(file_hash)
                    return response
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except Exception as e:
                        logging.error(f"Error removing temporary file: {str(e)}")
            
            except Exception as e:
                logging.error(f"File processing error: {str(e)}")
                return f"Error processing file: {str(e)}"
        else:
            return "File already processed"

rag_app = ChainlitRagApp()

@cl.on_chat_start
async def start():
    await cl.Message(content="RAG Assistant is ready!").send()
    upload_action = cl.Action(
        name="upload_docs",
        label="📄 Upload Documents",
        value="upload_docs"
    )
    clear_action = cl.Action(
        name="clear_docs",
        label="🗑️ Clear Documents",
        value="clear_docs"
    )

    await cl.Message(
        content="Choose an action:",
        actions=[upload_action, clear_action]
    ).send()

@cl.action_callback("upload_docs")
async def upload_docs_callback():
    files = await cl.AskFileMessage(
        content="Please upload PDF documents for analysis.",
        accept=["application/pdf"],
        max_size_mb=10,
        max_files=5
    ).send()

    for file in files:
        result = await rag_app.process_file(file)
        await cl.Message(content=result).send()

@cl.action_callback("clear_docs")
async def clear_docs_callback():
    rag_app.processed_files.clear()
    rag_app.setup_rag_client()
    await cl.Message(content="Cleared all processed documents. You can now upload new documents.").send()


@cl.on_message
async def main(message: cl.Message):
    if not rag_app.processed_files:
        await cl.Message(content="⚠️ No documents uploaded yet. Please upload PDFs first.").send()
        return

    response = rag_app.client.rag_chain.invoke(message.content)
    answer = response["answer"]
    sources = response.get("source_documents", [])

    # Send the response
    await cl.Message(content=answer).send()

    # Send the sources if any
    if sources:
        sources_text = "**Sources:**\n" + "\n".join([
            f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}..."
            for doc in sources
        ])
        await cl.Message(content=sources_text).send()

if __name__ == "__main__":
    cl.run()
