# -------------------------------
# 🛍️ MegaStore AI Assistant (Stable Streamlit Version - Fixed)
# -------------------------------

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
# تحميل البيانات وبناء السلسلة
# -------------------------------
@st.cache_resource
def load_chain():
    try:
        file_path = "megastore_dataset.txt"

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        # تقسيم النص
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        chunks = splitter.split_text(data)

        # بناء embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # بناء قاعدة البيانات الشعاعية
        vector_db = FAISS.from_texts(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # نموذج الإجابة
        qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=100,
            temperature=0.2,
            device=-1
        )
        llm = HuggingFacePipeline(pipeline=qa_pipeline)

        # الذاكرة
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # بناء سلسلة الأسئلة والأجوبة
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False
        )

        return qa_chain
    except Exception as e:
        st.error(f"⚠️ Error while loading chain: {e}")
        return None


qa = load_chain()

# -------------------------------
# واجهة المستخدم (الأسئلة والأجوبة)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Your Question:", placeholder="e.g. What services does MegaStore provide?")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        if qa is None:
            answer_text = "⚠️ Model failed to load. Please check the logs."
        else:
            try:
                answer_text = qa.run(user_input)
            except Exception as e:
                answer_text = f"⚠️ Error: {e}"

        st.session_state["messages"].append((user_input, answer_text))

# عرض المحادثة
for question, answer in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")
