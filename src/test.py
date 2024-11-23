from dotenv import load_dotenv

from lib import RagClient

load_dotenv(override=True)


def main():
    client = RagClient()
    #client.create_collection()
    client.add_documents()
    client.init_retriever()
    client.setup_chain()

    while True:
        query = input("Type your query: ")
        print(client.rag_chain.invoke(query)["answer"])


if __name__ == "__main__":
    main()
