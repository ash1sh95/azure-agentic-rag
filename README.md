# Azure Agentic RAG Pipeline

End-to-end **multi-agent RAG system** powered by **Azure OpenAI** and **LangChain**.  
Simulates real-world enterprise AI: **query routing → retrieval → generation**.

## Features
- Agentic workflow with **tool-calling**
- Vector store with **FAISS** (local) or **Azure Cosmos DB** (prod-ready)
- Deployable to **Azure Functions** or **App Service**
- Zero-cost demo using Azure free tier

## Tech Stack
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoft-azure)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)

## Quick Start

```bash
pip install -r requirements.txt
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
python rag_agent.py
