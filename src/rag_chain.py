
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def create_rag_chain(retriever):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an AI document assistant.

    Answer the question using ONLY the provided context.

    If the answer is not present in the context,
    clearly say that you could not find the answer
    in the uploaded documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
