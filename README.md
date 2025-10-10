# 🛍️ MegaStore AI Assistant

## 🧠 Project Overview
**MegaStore AI Assistant** is an intelligent, interactive chatbot built to simulate an AI-powered customer support system for an online fashion and apparel store.  
It allows users to ask natural language questions about MegaStore’s products, collections, delivery services, payment options, and more — and get accurate, context-aware responses in real time.  

The assistant demonstrates how modern e-commerce businesses can integrate **AI** and **RAG (Retrieval-Augmented Generation)** to provide automated, intelligent customer experiences — powered entirely by **open-source tools**.

---

## 🧩 Key Features
- 💬 **Conversational AI** – remembers chat history and continues conversations naturally.  
- 🧠 **RAG-based QA System** – combines LLM reasoning with document-based knowledge from a custom dataset (`megastore_dataset.txt`).  
- 🛍️ **E-commerce Knowledge Base** – trained on MegaStore’s detailed catalog (Men, Women, Kids, and Seasonal collections).  
- ⚡ **Free & Local** – uses open-source Hugging Face models (no paid API keys required).  
- 🖥️ **Interactive UI** – Streamlit interface with chat-style interaction.  
- ☁️ **Deployable on Streamlit Cloud** – easy one-click deployment from GitHub.

---

## 🧰 Tech Stack & Libraries Used

| Category | Tools & Libraries |
|-----------|------------------|
| Framework | 🧩 Streamlit |
| AI / NLP  | 🤗 Hugging Face Transformers, LangChain |
| Embeddings | 🧠 sentence-transformers / all-MiniLM-L6-v2 |
| Vector Database | 🗂️ FAISS |
| Language Model | 🗣️ FLAN-T5 (google/flan-t5-base) |
| Memory & Context | 💾 LangChain ConversationBufferMemory |
| Backend | ⚙️ Python 3.10+ |
| Deployment | ☁️ Streamlit Cloud / GitHub |

---

## 🚀 How It Works
1. Loads product and company data from `megastore_dataset.txt`.  
2. Converts the text into vector embeddings using **HuggingFaceEmbeddings**.  
3. Stores and retrieves information using a **FAISS** vector database.  
4. Uses **FLAN-T5** (Hugging Face model) for natural, context-aware text generation.  
5. Maintains conversation memory with **ConversationBufferMemory** for follow-up questions.  
6. Displays the full chat experience via **Streamlit’s chat interface**.

---

## 🌐 Use Cases
- AI customer support simulation for e-commerce  
- RAG-based chatbot demonstration  
- Educational project for LangChain + Hugging Face integration  
- Portfolio project for AI/ML engineers  

---

## ⚙️ Installation & Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/megastore-ai-assistant.git
cd megastore-ai-assistant
