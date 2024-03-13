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
import datasets
import re
import sys
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
sys.path.append('..')
from llm_recipes.utils import LLMSampleCB


wandb.login()

def prompt_formatter(row, mode='train'):
    if mode == 'train':
        return (
            f"""<human>: Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
{row["target"]}
<bot>:
{row["meaning_representation"]}"""
            )
    else:
        row["meaning_representation"] = ''
        return (
            f"""<human>: Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
{row["target"]}
<bot>:
{row["meaning_representation"]}"""
            )


def create_formatted_prompt(row, mode='train'):
    return prompt_formatter(row, mode)

data = datasets.load_dataset('GEM/viggo')
train_dataset = data["train"]
eval_dataset = data["validation"]

print(train_dataset)
print(eval_dataset)

# model_id = 'Open-Orca/Mistral-7B-OpenOrca'
model_id ="llmware/dragon-mistral-7b-v0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer = AutoTokenizer.from_pretrained(model_id)


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
    lora_dropout=0.05, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"], # the name of the layers to add LoRA
    # target_modules = ["model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.k_proj", "model.layers.31.self_attn.v_proj",
    #                   "model.layers.31.self_attn.o_proj", "model.layers.30.self_attn.q_proj", "model.layers.30.self_attn.k_proj", 
    #                   "model.layers.30.self_attn.v_proj", "model.layers.30.self_attn.o_proj",]
)

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



output_dir = "results_Viggo/"
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="finetune_for_viggo_dataset",
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