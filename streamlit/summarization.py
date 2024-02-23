import math
import time
import nltk
nltk.download('punkt')

import torch

from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

def generate_summary(input_text: str, device: str='cpu') -> str:
    """
    Generate summary from Arxiv research article.

    """
    # Start clock
    start_time = time.time()

    # Set model and torch method
    model_name = [
                "sambydlo/bart-large-scientific-lay-summarisation",
                "haining/scientific_abstract_simplification",
                "philschmid/bart-large-cnn-samsum"
            ]
    device = torch.device(device)
    tokenizer = BartTokenizer.from_pretrained(model_name[0])
    model = BartForConditionalGeneration.from_pretrained(model_name[0]).to(device)

    try:
        # Check length of article and batch if necessary
        batches = batch_input_text(input_text)
        # Summarize batches
        sub_summaries = []
        for i in range(len(batches)):
            output = model.generate(input_ids=tokenizer.encode(batches[i], return_tensors="pt").to(device),
                                    max_length=350)
            summary = tokenizer.decode(output[0])
            sub_summaries.append(summary)
        
        # Combine child summaries & remove specified separators
        joined_summaries = " ".join(sub_summaries)
        parent_input = joined_summaries.replace("</s><s><s><s>", "").replace("</s>", "")

        # Obtain final summary
        output = model.generate(input_ids=tokenizer.encode(parent_input, return_tensors="pt").to(device),
                                max_length=350)
        summary = tokenizer.decode(output[0])
        summary = summary.replace("</s><s><s><s>", "").replace("</s>", "")
    except:
        pass

    # Elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")

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

