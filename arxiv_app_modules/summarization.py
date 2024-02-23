import math
import time
import nltk
nltk.download('punkt')

import streamlit as st

import torch

from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

import requests



def query(payload, API_KEY):
    API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def generate_summary(input_text: str, API_KEY: str, device: str='cpu') -> str:
    """
    Generate summary from Arxiv research article.

    """
    # Start clock
    start_time = time.time()
    
    # Check length of article and batch if necessary
    batches = batch_input_text(input_text)
    if len(batches) >= 12:
        batches = batches[:12]
    st.write(f"Processing PDF in {len(batches)} batches...")

    # Summarize batches
    sub_summaries = []
    for i in range(len(batches)):
        st.write(f"Reading and processing batch {i+1}")
        try:
            output = query(payload={"inputs": batches[i]}, API_KEY=API_KEY)
            summary = output[0]['summary_text']
            sub_summaries.append(summary)
            st.caption(f'{summary.replace("</s><s><s><s>", "").replace("</s>", "")}')
            time.sleep(4)
        except:
            pass
    st.write("Final processing...")
    # Combine child summaries & remove specified separators
    joined_summaries = " ".join(sub_summaries)
    # Obtain final summary
    output = query(payload={'inputs': joined_summaries}, API_KEY=API_KEY)
    summary = output[0]['summary_text']

    # Elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Article summarized! Elapsed time: {int(elapsed_time)} seconds")

    return summary


def batch_input_text(input_text: str, batch_size: int=819) -> list:
    """
    :param input_text: str, research paper in full
    :param batch_size: int, 80% of max input tokens 
    :return: list, batched input text
    """
    try:
        tokens = nltk.word_tokenize(input_text)
        n_batches = math.ceil(len(tokens) / batch_size)

        if len(tokens) > batch_size:
            batches = [" ".join(tokens[(i * batch_size):((i + 1) * batch_size)]) 
                    for i in range(n_batches)]
        else:
            batches = [" ".join(tokens)]
    except Exception as e:
        raise e

    return batches  