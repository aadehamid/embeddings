# %%
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from rich import print
import torch
from typing import List, Tuple, Union, Callable
import numpy as np


from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import openai
import os

from dotenv import load_dotenv
load_dotenv()

from tenacity import retry, stop_after_attempt, wait_random_exponential


import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.api_core import retry as gretry


# %%
# openai.api_key = os.getenv('OPENAI_API_KEY')
# API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=API_KEY)
# %%
# Sentence embedding with bert
# define model
model_name = "bert-base-uncased"

# define the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
def bert_sentence_embed(input_sentence: str, model: BertModel = bert_model, word_ave: bool = True) -> Tensor:
    """A function to generate sentence embedding using Bert

    Args:
        input_sentence (str): The sentence to embed

    Returns:
        Tensor: Tensor output for the sentence embedding
    """

    temp_list = []
    for text in input_sentence:
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        # input_ids = tokenizer.encode(input_text_lst_news[0], return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)

            if word_ave:
                sentence_embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                sentence_embedding = outputs.last_hidden_state
                sentence_embedding = sentence_embedding[0, 0, :].reshape(1, -1)
            temp_list.append(sentence_embedding)
    concatenated_sent_tensor = torch.cat(temp_list)
    # # Convert the concatenated tensor to a numpy array
    sent_numpy_array = concatenated_sent_tensor.numpy()

    # # Reshape the numpy array into a 1 by 768 array
    sent_numpy_array = sent_numpy_array.reshape(len(temp_list), temp_list[0].shape[1])
    return sent_numpy_array

# %%
# A function to create bert and other embeddings
def create_sentence_embedding(input_text: List[str],model:Union[Callable, object], 
bert: bool = True, word_ave: bool = True) -> np.array:
    """A function to create sentence embedding from a list of text using 
        Bert or other open source model
    Args:
        input_text (List): List of sentence to create embedding for

    Returns:
        np.array: np.array for all of the the sentence embedding
    """
    embed_list = []
    # model_name = 'bert-base-uncased'
    if bert:
        sent_numpy_array = bert_sentence_embed(input_text_lst_news, model, word_ave)
        
    else:
        for text in input_text_lst_news:
            sen_emb = model.encode(text)
            embed_list.append(sen_emb)
        concatenated_sent_tensor = np.concatenate(embed_list)

        # Reshape the numpy array into a 1 by 768 array
        sent_numpy_array = concatenated_sent_tensor.reshape(len(embed_list), len(embed_list[0]))

    # Perform PCA for 2D visualization
    # convert the 768-dimensional array to 2-dimentional array for plotting purpose
    PCA_model = PCA(n_components=2)
    PCA_model.fit(sent_numpy_array)
    sent_low_dim_array = PCA_model.transform(sent_numpy_array)

    return sent_numpy_array, sent_low_dim_array

# %%
# A fucntion to compute cosine similarity
def compare(embeddings: np.array, idx1: int, idx2: int) -> float:
    """A function to compute cosine similarity between two embedding vectors

    Args:
        embeddings (np.array): An array of embeddings
        idx1 (int): Index of the first embedding 
        idx2 (int): index of the second embedding

    Returns:
        int: The distance between the two embeddings
    """
    item1 = embeddings[idx1, :].reshape(1, -1)
    item2 = embeddings[idx2, :].reshape(1, -1)
    distance = cosine_similarity(item1, item2)
    return distance.item()

# %%

#-----------------------------------------------
# A fucntion to compute sentence embedding using OpenAI
#-----------------------------------------------

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-similarity-davinci-001", **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = openai.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: List[str], model="text-similarity-babbage-001", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.embeddings.create(input=list_of_text, model=model, **kwargs).data
    embed_list = [d.embedding for d in data]
    embed_array = np.array(embed_list)

    # Convert to 2-dimensional vector to be able to visualize the embeddings
    PCA_model = PCA(n_components=2)
    PCA_model.fit(embed_array)
    sent_low_dim_array = PCA_model.transform(embed_array)
    return embed_array, sent_low_dim_array

    # embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# %%

#-----------------------------------------------
# A fucntion to compute sentence embedding using Google's Gemini
#-----------------------------------------------

def gembed_fn(model, input_text: List[str]) -> np.array:
    embed_list = []
    for text in input_text:
        # set the task type to semantic_similarity
        embedding = genai.embed_content(model = model, content = text, 
        task_type = "semantic_similarity")["embedding"]
        embed_list.append(embedding)
    embed_array = np.array(embed_list)
     # Convert to 2-dimensional vector to be able to visualize the embeddings
    PCA_model = PCA(n_components=2)
    PCA_model.fit(embed_array)
    sent_low_dim_array = PCA_model.transform(embed_array)
    return embed_array, sent_low_dim_array