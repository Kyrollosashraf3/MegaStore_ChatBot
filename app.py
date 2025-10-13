import sys
!{sys.executable} -m pip install langchain-community==0.2.12

#  Import

import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# إعداد واجهة Streamlit
# -------------------------------
st.set_page_config(page_title="🛍️ MegaStore AI Assistant", page_icon="🛒", layout="centered")
st.title("🛍️ MegaStore AI Assistant")
st.write("Welcome! Chat with MegaStore’s AI to learn more about our products and services.")

# -------------------------------
# تحميل الداتا من الملف
# -------------------------------

def load_chain():

    # Read data
    file_path = "data/megastore_dataset.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    # تقطيع النصوص
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    chunks = splitter.split_text(data)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # vector_data_base
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    # Retrievers
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_texts(chunks)
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6])
    
    # pipeline 
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=50)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    # chat_history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    # RetrievalChain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=faiss_retriever,  # or hybrid_retriever
        memory=memory
    )

qa = load_chain()


# session memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# textbox
user_input = st.text_input("Your Question:", placeholder="e.g. What services does MegaStore provide?")

if st.button("Ask") and user_input:
    answer = qa.invoke(user_input)
    st.session_state["messages"].append((user_input, answer["answer"]))

# عرض المحادثة
for question, answer in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")


