from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of our RAG execution graph"""
    messages : Annotated[Sequence[BaseMessage], add_messages]
    documents : List[Document]
    query : str
    rewritten_query : str
    answer : str
    generation_status: str # To track if we have content to generate
    
    