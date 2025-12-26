"""
The Brain (LangGraph): Contains the core decision logic (Rewrite -> Retrieve -> Grade -> Generate).
"""
import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from app.config import config
from app.core.state import AgentState
from app.services.vector_store import vectorstore

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model= config.MODEL_NAME,
    api_key= config.OPENROUTER_API_KEY,
    base_url= config.BASE_URL,
    temperature= 0.1
)

# -- Nodes --
async def rewrite_query(state: AgentState):
    """Optime the user query for semantic retrieval"""
    original_query = state['messages'][-1].content
    sysetm_prompt = """You are expert at optimizing queries for vector search.
    Look at the user's question and rewrite it to be precise , remove conversational fluff.
    Return ONLY the rewritten query string.
    """
    
    msg  = [
        SystemMessage(sysetm_prompt),
        HumanMessage(content=f"Original Query: {original_query}")
    ]
    
    response = await llm.ainvoke(msg)
    return {"rewritten_query": response.content}

async def retrieve_document(state: AgentState):
    """Retireve docs based on  rewritten query"""
    q = state["rewritten_query"]
    logger.info(f"Retrieveing for: {q}")
    
    # Retrieve slightly more docs to allow for filtering
    docs = vectorstore.similarity_search(query=q, k= 5)
    return {"documents" : docs}

async def grade_document(state: AgentState):
    """Filter out irrelevant documents to prevent hallucination"""
    query = state['rewritten_query']
    documents = state['documents']
    relevant_docs= []
    
    grade_system = """You are a strict relevance grader.
    Does the document contain information directly related to the query?
    Answer only 'yes' or 'no'
    """
    for doc in documents:
        msg = [
            SystemMessage(grade_system),
            HumanMessage(f"Query: {query}\n\nDocument snippet: {doc.page_content[:400]}")
        ]
        res = await llm.ainvoke(msg)
        if "yes" in res.content.lower():
            relevant_docs.append(doc)
 
    
    return{
        "documents": relevant_docs,
        "generation_status" : "go" if relevant_docs else "stop"
    }

async def generate_answer(state: AgentState):
    """Generate the final FAQ answer"""
    query = state["messages"][-1].content
    documents = state["documents"]
    
    context_str = "\n\n".join([f"Source ({d.metadata.get('filename')}): {d.page_content}" for d in documents])
    prompt = f"""You are helpful FAQ assistant. Use the provided context to answer the user's question.
    Guidelines:
    1. Answer Concisely and directly.
    2. Do not use outsie knowledge. If the answer isn't in the context, say so.
    3. Use professional tone.
    
    context:
    {context_str}
    
    User Question: {query}
    """
    
    response = await llm.ainvoke([HumanMessage(prompt)])
    return {"answer": response.content}


async def fallback_answer(state: AgentState):
    """Return a standard response when no info is found"""
    return{"answer":  "I'm sorry, but I couldn't find any information regarding that in the uploaded documents."}


# --- Edges ---
def decide_to_generate(state: AgentState) -> Literal["generate", "fallback"] :
    if state["generation_status"] == "go":
        return "generate"
    return "fallback" 


# --- Graph Construction ---
def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("rewrite", rewrite_query)   
    graph.add_node("retrieve", retrieve_document)   
    graph.add_node("grade", grade_document)   
    graph.add_node("generate", generate_answer)   
    graph.add_node("fallback", fallback_answer)
    
    graph.add_edge(START, 'rewrite')
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    
    graph.add_conditional_edges(
        "grade", 
        decide_to_generate,
        {
            "generate" : "generate",
            "fallback" : "fallback"
        }
    )
    
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    
    return graph.compile()

rag_graph = build_graph()