
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model

VECTOR_PATH = "vectorstore/faiss_index"


def create_vector_store(documents):

    embeddings = get_embedding_model()

    vector_db = FAISS.from_documents(
        documents,
        embeddings
    )

    vector_db.save_local(VECTOR_PATH)

    return vector_db


def load_vector_store():

    embeddings = get_embedding_model()

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
