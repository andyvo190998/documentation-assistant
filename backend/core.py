import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# embedding
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# vectorstore
vectorstore = PineconeVectorStore(
    index_name="documentation-assistant", embedding=embeddings
)

# init chat model
model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)  # k is max numbers of docs

    # Serialize docs
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer user query using retrieved documentation

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # build messages list
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs,
    }

if __name__ == "__main__":
    result = run_llm(query="what are deep agents")
    print(result)