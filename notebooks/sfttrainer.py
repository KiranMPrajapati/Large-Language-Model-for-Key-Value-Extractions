import json
import random
from numpy import dsplit
from numpy import dsplit
import wandb
from wandb import Api
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import default_data_collator, TrainingArguments
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import os
sys.path.append(Path(os.getcwd()).parent)
# from llm_recipes.utils import LLMSampleCB


wandb.login()

csv_file = pd.read_csv('index_info.csv')
index_info = csv_file['Index of data without headers'].tolist()

# Function to add backticks to HTML tags
def add_backticks_to_html_tags(html_content):
    # Define the regular expression pattern to match HTML tags
    pattern = r'<([^>]+)>'


    # Replace HTML tags with backticks added
    return re.sub(pattern, r'`<\1>`', html_content)


# row = alpaca[232]
# print(prompt_input(row))
def prompt_formatter(row, mode='train'):
    addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully. Provide the exact answer only."
    # return (
    #     f"""<s> [INST] {addon_prompt}\n
    #     {row['json_encoding']}\n
    #     {row['question']}\n
    #     [/INST]
    #     {row['answer']} </s>
    #     """
    #   )
    return (
        f"<human>: {addon_prompt}\n {row['json_encoding']}\n {row['question']}\n <bot>: {row['answer']}"
        )


def create_formatted_prompt(row, mode='train'):
    return prompt_formatter(row, mode)
# api = Api()
# artifact = api.artifact('spygaurad/alpaca_gpt4_dolly_ft/alpaca_gpt4_dolly_splitted', type='dataset')
# dataset_dir = artifact.download()
# ds = load_dataset("json", data_files='../data/convfinqa_json_max_length_2048.json')
# print(ds)
# train_dataset = ds["train"].select([i for i in range(15000)])
# print(train_dataset)
# eval_dataset = ds["train"].select([i for i in range(15000, 16193)])
# print(eval_dataset)

df = pd.read_csv('../data/pseudo_json_dataset_V2_2048.csv')
print(df.shape)
train_df = df[:15000]
train_df["value"] = (
    "<human>:" +
    train_df['pseudo_json_encoding'] + "\n" +
    train_df['question'] + "\n" +
    "<bot>:" +
    train_df['answer']
)
train_df = train_df.dropna()

custom_train_ds = pd.DataFrame()
custom_train_ds["prompt"] = train_df['value']

train_dataset = Dataset.from_pandas(custom_train_ds)
print(train_dataset)

test_df = df[15000:]
test_df["value"] = (
    "<human>:" +
    test_df['pseudo_json_encoding'] + "\n" +
    test_df['question'] + "\n" +
    "<bot>:"
)
test_df = test_df.dropna()

custom_test_ds = pd.DataFrame()
custom_test_ds["prompt"] = test_df['value']

eval_dataset = Dataset.from_pandas(custom_test_ds)
print(eval_dataset)

# model_id = 'Open-Orca/Mistral-7B-OpenOrca'
model_id ="llmware/dragon-mistral-7b-v0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)


model_kwargs = dict(
    device_map={"" : 0},
    trust_remote_code=True,
    # low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
    use_cache=False,
)




peft_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.5, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    # target_modules = ['fc1', 'fc2', 'Wqkv', 'out_proj'],# Specific for Phi2
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
    # target_modules = ["model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.k_proj", "model.layers.31.self_attn.v_proj",
    #                   "model.layers.31.self_attn.o_proj", "model.layers.30.self_attn.q_proj", "model.layers.30.self_attn.k_proj", 
    #                   "model.layers.30.self_attn.v_proj", "model.layers.30.self_attn.o_proj",]
)


batch_size = 1
# 3 * (4 * 32)
<<<<<<< HEAD
num_train_epochs = 2
gradient_accumulation_steps = 128
=======
num_train_epochs = 3
gradient_accumulation_steps = 32
>>>>>>> c2d6a8612ff84defe4a97cbe34017c78381bb411
total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)

output_dir = "results_V3/"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
    max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    # evaluation_strategy="steps",
    # eval_steps=20,
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=20,
)

trainer = SFTTrainer(
    model=model_id,
    model_init_kwargs=model_kwargs,
    train_dataset=train_dataset,
    # eval_dataset = eval_dataset,
    packing=True,
    max_seq_length=1024,
    args=training_args,
    formatting_func=create_translation_prompt,
    peft_config=peft_config,
)


# remove answers
# # def create_prompt_no_anwer(row):
    # # return {"text": create_translation_prompt(row, mode='eval')}

# 
# test_dataset = eval_dataset.map(create_prompt_no_anwer)
# wandb_callback = LLMSampleCB(trainer, eval_dataset, num_samples=10, max_new_tokens=256)
# trainer.add_callback(wandb_callback)


trainer.train()
wandb.finish()
