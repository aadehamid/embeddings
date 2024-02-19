from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from pathlib import Path


def get_index(data, index_name, embed_model, chromapath):
    index = None
    db = chromadb.PersistentClient(path=chromapath)
    chroma_collection = db.get_or_create_collection(index_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if not os.path.exists(chromapath):
        print("building index", index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(data, storage_context=storage_context, embed_model=embed_model,
                                                show_progress=True)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store,
                                                   embed_model=embed_model, )

    return index


embedding_model_name = "text-embedding-3-large"
embed_model = OpenAIEmbedding(model=embedding_model_name)


canada_filepath = Path.joinpath(Path.cwd(), "RAG", "agent", 'agent_data', 'Canada.pdf')
chromapath = str(
    Path.joinpath(Path.cwd().parent, 'agent_data', 'agent_chroma_db'))
canada_pdf = PDFReader().load_data(file=canada_filepath)
canada_index = get_index(canada_pdf, "canada", embed_model,
                         chromapath)
canada_engine = canada_index.as_query_engine()
