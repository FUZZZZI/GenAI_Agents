## Utils
import os, sys, typer   
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

## Tools/Agent
from phi.assistant import Assistant #Depriciated
from phi.agent import Agent

## Local storage
from phi.storage.assistant.postgres import PgAssistantStorage #Depriciated
from phi.storage.agent.postgres import PgAgentStorage

## Data reader
from phi.knowledge.pdf import PDFUrlKnowledgeBase

## VectorDB
from phi.vectordb.pgvector import PgVector, SearchType

## LLM
from phi.model.groq import Groq



# from sentence_transformers import SentenceTransformer
# import torch
# from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
# embeddings = SentenceTransformerEmbedder().get_embedding("The quick brown fox jumps over the lazy dog.")

# # Print the embeddings and their dimensions
# print(f"Embeddings: {embeddings[:5]}")
# print(f"Dimensions: {len(embeddings)}")
# os.environ["PGVECTOR_VECTOR_SIZE"] = str(384)



os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url,)
                        # embedder=SentenceTransformerEmbedder()),
)
knowledge_base.load()

storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(pdf_assistant)