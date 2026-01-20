import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import faiss
from uuid import uuid4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# Load env variables
load_dotenv()

# ------------------ PDF PROCESSING ------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def build_vectorstore(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    if not raw_text.strip():
        raise ValueError("No text extracted from PDFs")

    chunks = get_text_chunks(raw_text)

    # Create Documents
    documents = [
        Document(page_content=chunk, metadata={"source": "pdf"})
        for chunk in chunks
    ]
    ids = [str(uuid4()) for _ in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # ---- NEW FAISS INITIALIZATION (OFFICIAL WAY) ----
    dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vectorstore.add_documents(documents=documents, ids=ids)

    return vectorstore


# ------------------ AGENT ------------------

def build_agent(vectorstore):
    model = init_chat_model("gpt-4o")


    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant context from PDF vector store."""
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant context found."

        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )

    system_prompt = (
        "You are a helpful assistant answering questions using retrieved "
        "context from a PDF knowledge base. "
        "Always use the retrieval tool for factual questions. "
        "If the answer is not present, say you don't know."
    )

    
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )
    
    return agent


# ------------------ STREAMLIT UI ------------------

def main():
    st.set_page_config(page_title="Chat with PDF (Agent + FAISS)", page_icon="📚")
    st.title("📚 Chat with your PDF (Agent-based RAG)")

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        st.header("⚙️ Configuration")

        if os.getenv("OPENAI_API_KEY"):
            st.success("OpenAI API Key loaded")
        else:
            st.error("OPENAI_API_KEY missing")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    try:
                        vectorstore = build_vectorstore(pdf_docs)
                        st.session_state.agent = build_agent(vectorstore)
                        st.session_state.processComplete = True
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(str(e))

    # Chat history
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # Chat input
    if st.session_state.processComplete:
        if prompt := st.chat_input("Ask a question about your PDFs"):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    events = st.session_state.agent.stream(
                        {"messages": st.session_state.messages},
                        stream_mode="values"
                    )

                    final_answer = ""
                    for event in events:
                        final_answer = event["messages"][-1].content

                    st.write(final_answer)
                    st.session_state.messages.append(
                        AIMessage(content=final_answer)
                    )
    else:
        st.info("👈 Upload PDFs and click Process")


if __name__ == "__main__":
    main()
