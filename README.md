# ♻️ EcoSort AI: Intelligent Municipal Waste Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![IBM WatsonX](https://img.shields.io/badge/IBM_WatsonX-Granite_Model-0F62FE.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG_Architecture-1C3C3C.svg)

EcoSort AI is a Retrieval-Augmented Generation (RAG) powered web application designed to help citizens and businesses navigate complex municipal waste segregation policies. Built for the **1M1B & IBM SkillsBuild AI for Sustainability Internship**.

## 🎥 Project Demo

https://github.com/user-attachments/assets/37c7f13a-be6e-4682-885a-151ed83ae4e2


## 🌍 The Problem & SDG Alignment
This project aligns with **SDG 11 (Sustainable Cities)** and **SDG 12 (Responsible Consumption)**. 
Lengthy municipal policy PDFs cause confusion regarding the disposal of e-waste, hazardous materials, and construction debris. This leads to improper segregation, contaminated recycling streams, and overflowing landfills. 

## 💡 The Solution
EcoSort AI bridges the gap between citizens and municipal guidelines. Users can ask questions in natural language, and the application instantly retrieves the exact rule, penalty, or disposal method directly from the official uploaded policy document, ensuring zero AI hallucination.

## 🛠️ Tech Stack & Architecture
* **LLM / Orchestration:** IBM Granite (`ibm/granite-4-h-small`) accessed via IBM WatsonX API.
* **Architecture:** Retrieval-Augmented Generation (RAG).
* **Framework:** LangChain (PyPDFLoader, Text Splitters, RetrievalQA).
* **Vector Store & Embeddings:** FAISS Vector Database with HuggingFace Sentence-Transformers.
* **Frontend:** Streamlit.
* **System Design:** IBM BOB.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1032241501-art/ecosort-ai-assistant.git
   cd ecosort-ai-assistant
