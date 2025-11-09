import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🛍️ MegaStore AI Assistant", page_icon="🛒", layout="centered")
st.title("🛍️ MegaStore AI Assistant")
st.write("Welcome! Chat with MegaStore’s AI to learn about our products and services.")

# -------------------------------
# Load data and build embeddings
# -------------------------------
file_path = "megastore_dataset.txt"  # replace with your dataset path
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

# Split data into paragraphs manually
documents = [para for para in data.split("\n\n") if para.strip() != ""]

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.from_texts(documents, embeddings)

# -------------------------------
# Initialize LLM
# -------------------------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# -------------------------------
# User interface (Q&A)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Your Question:", placeholder="e.g. What services does MegaStore provide?")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        try:
            # Get top 3 relevant documents from FAISS
            docs = vector_db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in docs])

            # Prepare prompt for LLM
            prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{user_input}"

            # Run LLM
            answer_text = llm(prompt)
        except Exception as e:
            answer_text = f"⚠️ Error: {e}"

        # Save Q&A in session state
        st.session_state["messages"].append((user_input, answer_text))

# Display chat history
for question, answer in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")
