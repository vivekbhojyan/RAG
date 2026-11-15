
import streamlit as st
import os
import tempfile

from dotenv import load_dotenv

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG")
st.caption("Ask questions from your uploaded documents using LangChain RAG")

if not os.getenv("GROQ_API_KEY"):
    st.error("Please configure GROQ_API_KEY in your .env file.")
    st.stop()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):

    if not uploaded_files:
        st.sidebar.warning("Upload documents first.")

    else:
        all_documents = []

        with st.spinner("Processing documents..."):

            for uploaded_file in uploaded_files:

                suffix = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    documents = load_document(temp_path)

                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name

                    all_documents.extend(documents)

                finally:
                    os.remove(temp_path)

            if not all_documents:
                st.error("No readable text found.")
                st.stop()

            chunks = split_documents(all_documents)

            st.session_state.vector_db = create_vector_store(chunks)

            st.session_state.chat_history = []

        st.sidebar.success(
            f"Processed {len(all_documents)} pages and {len(chunks)} chunks!"
        )

st.divider()

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your documents...")

if question:

    if st.session_state.vector_db is None:
        st.warning("Please upload and process documents first.")

    else:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                retriever = get_retriever(
                    st.session_state.vector_db
                )

                docs = retriever.invoke(question)

                if not docs:
                    answer = "I couldn't find relevant information in your documents."
                    sources = []
                else:
                    rag_chain = create_rag_chain(retriever)
                    answer = rag_chain.invoke(question)
                    sources = docs

                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        for doc in sources:
                            st.write(
                                f"**Source:** {doc.metadata.get('source', 'Unknown')}"
                            )
                            page = doc.metadata.get("page")
                            st.write(
                                f"**Page:** {page + 1 if isinstance(page, int) else 'N/A'}"
                            )
                            st.write(doc.page_content)
                            st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
