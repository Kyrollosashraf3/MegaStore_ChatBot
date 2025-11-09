#from langchain.retrievers import EnsembleRetriever
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory  



from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


file_path = "/content/megastore_dataset.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()


#data = data.split("\n\n")

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
                      chunk_size=300,
                      chunk_overlap=100,
                      separators= ["\n\n", "\n", ".", "!", "?", ",", " "]
)

chunks  = splitter.split_text(data)





# 1- embeddings from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# sentence-transformers/all-MiniLM-L6-v2 - sentence-transformers/all-MiniLM-L12-v2"


# 2- We save the embeddings in the FAISS database.
vector_db = FAISS.from_texts(chunks, embeddings)


# 3-  retriever (FAISS- BM25Retriever)
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # k : n of the most similarity results
bm25_retriever = BM25Retriever.from_texts(chunks)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]  # weights of 2 ways
)






qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=20)
#other models : google/flan-t5-small , base , large, google/t5-v1_1-base

# LangChain
llm = HuggingFacePipeline(pipeline=qa_pipeline)


# memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=faiss_retriever,
    memory=memory
)




# -------------------------------
# Build the Q&A chain
# -------------------------------
try:
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
    qa_chain = create_retrieval_chain(
        history_retriever,
        llm,
        answer_prompt
    )

except Exception as e:
    st.error(f"⚠️ Error while loading chain: {e}")
    qa_chain = None

qa = qa_chain

# -------------------------------
# User interface (Q&A)
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
                # Add user question to memory
                memory.chat_memory.add_message(HumanMessage(content=user_input))
                # Run the chain
                output = qa.invoke({"input": user_input, "chat_history": memory.chat_memory.messages})
                answer_text = output["answer"]
            except Exception as e:
                answer_text = f"⚠️ Error: {e}"

        st.session_state["messages"].append((user_input, answer_text))

# Display the chat
for question, answer in st.session_state["messages"]:
    st.markdown(f"**🧍‍♂️ You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")


