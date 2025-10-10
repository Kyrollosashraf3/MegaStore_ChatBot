import sys
!{sys.executable} -m pip install langchain-community==0.2.12


import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline

# -------------------------------
# إعداد واجهة Streamlit
# -------------------------------
st.set_page_config(page_title="🛍️ MegaStore AI Assistant", page_icon="🛒", layout="centered")
st.title("🛍️ MegaStore AI Assistant")
st.write("Welcome! Chat with MegaStore’s AI to learn more about our products and services.")

# -------------------------------
# تحميل الداتا من الملف
# -------------------------------
@st.cache_data
def load_data():
    with open("megastore_dataset.txt", "r", encoding="utf-8") as f:
        return f.readlines()

data = load_data()

# -------------------------------
# إعداد Embeddings و Vector DB
# -------------------------------
@st.cache_resource
def create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(data, embeddings)

vector_db = create_vector_db()

# -------------------------------
# إعداد LLM و الـ Chain
# -------------------------------
@st.cache_resource
def create_conversational_chain():
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory
    )
    return chain

qa = create_conversational_chain()

# -------------------------------
# واجهة الدردشة
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال المستخدم
if question := st.chat_input("Type your question here..."):
    # أضف السؤال إلى المحادثة
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # احصل على الرد من الموديل
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa({"question": question})
            answer = result["answer"]
            st.markdown(answer)

    # احفظ الرد في الجلسة
    st.session_state.messages.append({"role": "assistant", "content": answer})
