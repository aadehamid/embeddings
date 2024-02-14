# %%
from dotenv import find_dotenv, load_dotenv
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from rich import print as pprint

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
from RAG.WO_notebooks.rag_utils  import (
    load_data_to_sql_db,
    text_to_query_engine,
    setup_query_engines,
    get_retry_guideline_response,
    evaluate_and_transform_query,
    load_index_from_weaviate,)

# -----------------------------------------------------------------------------
# llama_index imports
# --------------------------------------------------------------------------------
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings


# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-large"
# embedding_model_name="local:BAAI/bge-small-en-v1.5"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model

# create weaviate client
auth_config = AuthApiKey(api_key=WEAVIATE_APIKEY)

client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)

# %%
# load vector from weaviate
work_order_index = load_index_from_weaviate(client,"WorkOrderIndex" )

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
# path to the raw dataset
table_name = "work_order_table"
dbpath = "../data/wo_data.db"
wo_data_path = "/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/embeddings/RAG/WO_notebooks/data/excavator_2015_cleaned_forpdl.csv"

conn, engine, data, document = load_data_to_sql_db(wo_data_path, dbpath, table_name,
                                                   columns_to_embed, columns_to_metadata)
_, sql_database = text_to_query_engine(model_name, embedding_model_name, table_name, engine)
query_router_engine = setup_query_engines(sql_database, work_order_index, table_name)
response = get_retry_guideline_response(
    query_router_engine, "How much in total did we spend in 2011?", guideline=False
)
print(response)

