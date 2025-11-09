# -------------------------------
# 🛍️ MegaStore AI Assistant (New LangChain API - Streamlit Stable)
# -------------------------------

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.memory import ConversationBufferMemory

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🛍️ MegaStore AI Assistant", page_icon="🛒", layout="centered")
st.title("🛍️ MegaStore AI Assistant")
st.write("Welcome! Chat with MegaStore’s AI to learn about our products and services.")

# -------------------------------
# Load Chain
# -------------------------------
@st.cache_resource
def load_chain():

    # Load data
    file_path = "megastore_dataset.txt"
    data = open(file_path, "r", encoding="utf-8").read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(data)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vector_db = FAISS.from_texts(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # LLM
    t5 = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=200,
        temperature=0.2
    )
    llm = HuggingFacePipeline(pipeline=t5)

    # Memory
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # History-aware retriever
    history_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("system", "Use chat history to clarify the user question when needed.")
    ])

    history_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=history_prompt
    )

    # Response prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are MegaStore’s helpful AI assistant."),
        ("user", "{input}"),
        ("assistant", "Use the retrieved context:\n{context}")
    ])

    # Main chain
    chain = create_retrieval_chain(
        history_retriever,
        llm,
        answer_prompt
    )

    return chain, memory


chain, memory = load_chain()

# -------------------------------
# Chat UI
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Your Question:", placeholder="e.g. What services does MegaStore provide?")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):

        # Add message to memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Run chain
        try:
            output = chain.invoke({"input": user_input, "chat_history": memory.chat_memory.messages})
            answer = output["answer"]
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        st.session_state["messages"].append((user_input, answer))

# Show chat
for q, a in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
