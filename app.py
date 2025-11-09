import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🛍️ MegaStore AI Assistant", page_icon="🛒", layout="centered")
st.title("🛍️ MegaStore AI Assistant")
st.write("Welcome! Chat with MegaStore’s AI to learn about our products and services.")

# -------------------------------
# Load data and build embeddings
# -------------------------------
file_path = "/content/megastore_dataset.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)
chunks = splitter.split_text(data)

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.from_texts(chunks, embeddings)

# Use FAISS retriever only
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# Initialize LLM
# -------------------------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# -------------------------------
# Memory for chat history
# -------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------
# Build the QA chain
# -------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# -------------------------------
# User interface (Q&A)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Your Question:", placeholder="e.g. What services does MegaStore provide?")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        try:
            # Run the QA chain
            answer_text = qa.run(user_input)
        except Exception as e:
            answer_text = f"⚠️ Error: {e}"

        st.session_state["messages"].append((user_input, answer_text))

# Display chat messages
for question, answer in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")
