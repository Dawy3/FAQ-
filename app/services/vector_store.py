import logging
from pinecone import Pinecone ,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import config

logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = config.EMBEDDING_MDOEL,
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embedding': True}
)

def get_vectorstore():
    """Check if index exists, create if not, and return the store."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if config.INDEX_NAME not in existing_indexes:
        # Calculate dimesnion dynamically 
        dimension = len(embeddings.embed_query("test"))
        logger.info(f"Creating index {config.INDEX_NAME} with dimension {dimension}")
        pc.create_index(
            name= config.INDEX_NAME,
            dimension = dimension,
            metric= 'cosine',
            spec=ServerlessSpec(cloud='aws', region = "us-east-1")
        )
    
    return PineconeVectorStore(
        index_name= config.INDEX_NAME,
        embedding= embeddings
    )
    
# Add a function to get it on demand
_vectorstore = None

def get_active_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        # This will only run when the API actually needs it
        _vectorstore = get_vectorstore()
    return _vectorstore
        
    