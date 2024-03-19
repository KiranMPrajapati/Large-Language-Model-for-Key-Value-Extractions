import json
import random
from numpy import dsplit
from numpy import dsplit
import numpy as np
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
from transformers import EarlyStoppingCallback
import pandas as pd
import torchmetrics
from io import StringIO
import csv 

from bs4 import BeautifulSoup

import re
import sys
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
sys.path.append('..')
from llm_recipes.utils import LLMSampleCB

wandb.login()

def convert_to_csv(table):
    # Open StringIO object for writing
    csv_buffer = StringIO()


    # Write rows to the StringIO object
    writer = csv.writer(csv_buffer)
    trows = table.find_all('tr')
    for trow in trows:
        writer.writerow([cell.get_text() for cell in trow.find_all(['td', 'th'])])


    # Get the CSV data as a string
    csv_data = csv_buffer.getvalue()
    return csv_data

def prompt_formatter(row, mode='train'):
    soup = BeautifulSoup(row['html_content'], 'html.parser')

    # Find the table
    table = soup.find('table')
    csv_data = convert_to_csv(table)
    table_descriptions = "The csv contains financial data from an S&P 500 earnings."
    eval_mode = True
    if mode == 'train':
        eval_mode = False
        return (f"""[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].
    ### [Table Text]
    {table_descriptions}

    ### [Table]
    ```
    {csv_data}
    ```

    ### [Question]
    {row['question']}

    ### [Solution][/INST] {row['answer']}"""
        )
    if eval_mode:
        row['answer'] = ''
        return (
            f"""[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].
    ### [Table Text]
    {table_descriptions}

    ### [Table]
    ```
    {csv_data}
    ```

    ### [Question]
    {row['question']}

    ### [Solution][/INST] {row['answer']}"""
            )



def create_formatted_prompt(row, mode='train'):
    return prompt_formatter(row, mode)
# api = Api()
# artifact = api.artifact('spygaurad/alpaca_gpt4_dolly_ft/alpaca_gpt4_dolly_splitted', type='dataset')
# dataset_dir = artifact.download()
ds = load_dataset("json", data_files='../data/pseudo_json_dataset_V3_2048.json')
print(ds)
train_dataset = ds["train"].select([i for i in range(15000)])
print(train_dataset)
eval_dataset = ds["train"].select([i for i in range(15000, 17538)])
print(eval_dataset)

# model_id = 'Open-Orca/Mistral-7B-OpenOrca'
model_id ="RUCKBReasoning/TableLLM-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model_init_kwargs = dict(
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
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"], # the name of the layers to add LoRA
    
)

def peft_module_casting_to_bf16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.bfloat16)
        elif isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    model_id, **model_init_kwargs
)
model = get_peft_model(model, peft_config)
# peft_module_casting_to_bf16(model)

batch_size = 1
# 3 * (4 * 32)
num_train_epochs = 3
gradient_accumulation_steps = 128
total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)

grouped_params = model.parameters()
optimizer=torch.optim.Adam(grouped_params, lr=0.0003)
# lambda1 = lambda num_train_epochs: num_train_epochs // 30
# scheduler= torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, ])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
optimizers = optimizer, scheduler


output_dir = "results_V11/"
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="finetune_lm_head_transformer_layers_and_mlp_layers_TableLLM",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="steps",
    eval_steps=20,
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=20,
    # metric_for_best_model = 'f1',
    load_best_model_at_end=True,
)

def compute_metrics(p):  
    pred, groundtruth = p 
    pred = np.argmax(pred, axis=2)
    pred_flatten = pred.flatten()
    groundtruth_flatten = groundtruth.flatten()
    accuracy = accuracy_score(groundtruth_flatten, pred_flatten)
    precision = precision_score(groundtruth_flatten, pred_flatten, average='micro')
    recall = recall_score(groundtruth_flatten, pred_flatten, average='micro')
    f1 = f1_score(groundtruth_flatten, pred_flatten, average='micro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



trainer = SFTTrainer(
    model=model,
    # model_init_kwargs=model_kwargs,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset,
    packing=True,
    max_seq_length=2048,
    args=training_args,
    # dataset_text_field='prompt',
    formatting_func=create_formatted_prompt,
    # peft_config=peft_config,
    optimizers = (optimizer, scheduler)
    # compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
)


# remove answers
def create_prompt_no_anwer(row):
    return {"text": create_formatted_prompt(row, mode='eval')}


test_dataset = eval_dataset.map(create_prompt_no_anwer)
wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)


trainer.train()
