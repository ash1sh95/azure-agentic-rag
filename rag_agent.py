# rag_agent.py
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Config
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    temperature=0,
    max_tokens=500
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002"
)

# Sample Docs (replace with your own)
docs = [
    "Azure OpenAI enables secure, enterprise-grade LLM deployment.",
    "Agentic AI uses tools and reasoning to solve complex tasks.",
    "Databricks Unity Catalog provides governance for ML assets.",
    "RAG improves LLM accuracy by grounding responses in data."
]

vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()

@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context for the query."""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

tools = [retrieve_context]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Azure AI expert. Use tools to retrieve context, then answer concisely."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    response = agent_executor.invoke({
        "input": "How does Azure support agentic AI workflows?"
    })
    print("\nAnswer:", response["output"])
