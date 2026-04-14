import streamlit as st
from ollama import chat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# ================= LOAD & INDEX DATA =================
@st.cache_resource
def load_vectorstore():
    with open("itc_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ================= UI =================
st.title("ITC Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ================= CHAT =================
if prompt := st.chat_input("Ask about ITC..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 🔎 Retrieve relevant context
    docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in docs])

    # 🧠 Build RAG prompt
    rag_prompt = f"""
You are an assistant for ITC (Information Technology Club).

ONLY answer using the context below.
If the answer is not in the context, Try to respond in general like a normal chat person . 
If he changes his language or speak an other language, answer in that language .
Context:
{context}

Question:
{prompt}
"""

    # 🤖 Call Ollama
    response = chat(
        model="llama3",
        messages=[{"role": "user", "content": rag_prompt}],
        stream=False
    )

    assistant_msg = response["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").write(assistant_msg)