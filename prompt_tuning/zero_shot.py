import torch
import difflib
import torchmetrics
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    pipeline,
)

from tqdm import tqdm 
import pandas as pd
import numpy as np
import re

df = pd.read_csv("../data/pseudo_json_dataset_V3_2048.csv")

MODEL_NAME = "llmware/dragon-mistral-7b-v0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

generation_config.max_new_tokens = 512
generation_config.temperature = 0.0001
generation_config.do_sample = True

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    # streamer=streamer,
)

addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully. Your task is to accurately respond to the specific question asked."

def format_prompt(row):
    input_prompt = f"<human>: {addon_prompt}\n {row['pseudo_json_encoding']}\n {row['question']}\n <bot>: "
    return input_prompt

def extract_value_after_bot(text):
    # Extract text after <bot>:
    extracted_value = re.search(r'<bot>:\s*(.*)', text).group(1).strip()
    return extracted_value

output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction', 'diff_score', 'is_answer_correct'])

difflib_similarity = 0 
true_pos = 0 
partial_true_pos = 0 
for index, row in tqdm(df.iterrows()):
    prompt = format_prompt(row)
    output = llm(prompt)[0]['generated_text']
    pred = extract_value_after_bot(output)
    pred = str(pred).lstrip().replace('$', '').lstrip()
    ground_truth = str(row['answer']).lstrip().replace('$', '').lstrip()
    true_pos_index = 0 
    if pred == ground_truth:
        true_pos += 1 
        true_pos_index = 1 
    similarity_ratio = difflib.SequenceMatcher(None, pred, row['answer']).ratio()
    difflib_similarity += similarity_ratio
    output_df.at[index, 'index'] = df.loc[index, 'index']
    output_df.at[index, 'question'] = df.loc[index,'question']
    output_df.at[index, 'answer'] = ground_truth
    output_df.at[index, 'prediction'] = pred
    output_df.at[index, 'diff_score'] = similarity_ratio
    output_df.at[index, 'is_answer_correct'] = true_pos_index


similarity = difflib_similarity/len(df)
print('similarity', similarity)
accuracy = true_pos/len(df)
print('accuracy', accuracy)

output_df.to_csv('../output/pseudo_json_dataset_V3_2048_zero_shot_results.csv', index=False)


