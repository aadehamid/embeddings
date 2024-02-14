# %%
from dotenv import find_dotenv, load_dotenv
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from rich import print as pprint
import json

load_dotenv(find_dotenv())
import os

# -----------------------------------------------------------------------------
# openai and weaviate
import openai

openai.api_key = os.getenv("OPENA_AI_KEY")
import weaviate

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_APIKEY = os.getenv("WEAVIATE_API_KEY")
from weaviate.auth import AuthApiKey

# -----------------------------------------------------------------------------
# import utilities
from RAG.WO_notebooks.new_rag_utils  import (
    load_data_to_sql_db,
    text_to_query_engine,
    build_sentence_window_index_vector_DB,
    setup_query_engines,
    get_retry_guideline_response,
    evaluate_and_transform_query,
    build_sentence_window_index,
)

# from RAG.WO_notebooks.rag_utils import build_sentence_window_index_vector_DB


# -----------------------------------------------------------------------------
# llama_index imports
# --------------------------------------------------------------------------------
from llama_index.core.response.notebook_utils import display_response
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.weaviate import WeaviateVectorStore


# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-large"
# embedding_model_name="local:BAAI/bge-small-en-v1.5"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model

# %%
auth_config = AuthApiKey(api_key=WEAVIATE_APIKEY)

client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)

# %%
# # Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=WEAVIATE_URL,
#     auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_APIKEY))

# %% [markdown]
# # Load Work Order Data

# %%
# path to the raw dataset
table_name = "work_order_table"
dbpath = "../data/wo_data.db"
wo_data_path = "/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/embeddings/RAG/WO_notebooks/data/excavator_2015_cleaned_forpdl.csv"
conn, engine, data = load_data_to_sql_db(wo_data_path, dbpath, table_name)

# %% [markdown]
# # test the query engine

# %%
# query_engine,sql_database = text_to_query_engine(model_name, embedding_model_name, table_name, engine)
# # query_str = "Which work oder cost 183.05?"
# query_str = "How much in total did we spend in 2011?"
# response = query_engine.query(query_str)
# print(response)

# %% [markdown]
# # Create document object

# %%
columns_to_embed = [
    "OriginalShorttext",
    "MajorSystem",
    "Part",
    "Action",
    "FM",
    "Location",
    "FuncLocation",
]

columns_to_metadata = [
    "Asset",
    "Cost",
    "RunningTime",
    "Variant",
    "Comments",
    "SuspSugg",
    "Rule",
]


docs = []
for i, row in data.iterrows():
    to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
    values_to_embed = {k: str(row[k]) for k in columns_to_embed if k in row}
    to_embed = "\n".join(
        f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
    )
    newDoc = Document(text=to_embed, metadata=to_metadata)
    docs.append(newDoc)

# Create a single document from a list of Documents
# this is what we will chunk up and store with its embedding in Weaviate
document = Document(text="\n\n".join([doc.text for doc in docs]))

# %% [markdown]
# # Build Sentence window

# %%
# load work_order_schema from json
# client.schema.delete_class("WorkOrder")


with open("RAG/WO_notebooks/config/work_order_schema", "r") as f:
    work_order_schema = json.load(f)

client.schema.create(work_order_schema)
print("Product schema was created.")


# %%
# # Create index in weaviate
# work_order_index = build_sentence_window_index_vector_DB(
#     document=document,
#     client=client,
#     llm=llm,
#     embed_model=embed_model,
#     prefix="Work_order_sent_win_index",
# )
# Load vector index from weaviate
# "WorkOrderIndex"
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="WorkOrderIndex"
)

work_order_index = VectorStoreIndex.from_vector_store(vector_store)

# %% [markdown]
# # Save the index in vectore store

# %%
_, sql_database = text_to_query_engine(model_name, embedding_model_name, table_name, engine)
query_router_engine = setup_query_engines(sql_database, work_order_index, table_name)
response = get_retry_guideline_response(
    query_router_engine, "How much in total did we spend in 2011?", guideline=False
)
print(response)

# %%
# query_str = "Which work oder cost 183.05?"
# query_str = "How much in total did we spend in 2011?"
# response = query_router_engine.query(query_str)
# print(str(response))
# display_response(response)

# %% [markdown]
#

# %% [markdown]
# # Evaluating query and prompts
# %%
query_str = "How much in total did we spend in 2011?"
feedback, transformed_query = evaluate_and_transform_query(
    query_str, response, DEFAULT_GUIDELINES
)

# %%
print(transformed_query)

# %%

pprint(feedback)

# %%
