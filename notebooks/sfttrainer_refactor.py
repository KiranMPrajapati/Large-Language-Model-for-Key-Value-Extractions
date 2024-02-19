import wandb
from wandb import Api
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_cosine_schedule_with_warmup
from transformers import default_data_collator, TrainingArguments
from datasets import load_dataset
from torch import nn
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from bs4 import BeautifulSoup
import pandas as pd
import argparse
import re
# from llm_recipes.utils import LLMSampleCB

MODEL_ID = 'Open-Orca/Mistral-7B-OpenOrca'

wandb.login()

csv_file = pd.read_csv('../data/index_info.csv')
index_info = csv_file['Index of data without headers'].tolist()

# Function to add backticks to HTML tags
def add_backticks_to_html_tags(html_content):
    # Define the regular expression pattern to match HTML tags
    pattern = r'<([^>]+)>'
    return re.sub(pattern, r'`<\1>`', html_content)

# remove answers
def create_prompt_no_anwer(row):
    return {"text": create_translation_prompt(row, mode='eval')}
    
def prompt_translation(row, mode='train'):
    if mode == 'train':
        addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
        input_prompt =f"""<s> [INST] {addon_prompt}\n
            {row['json_encoding']}\n
            {row['question']}\n
            [/INST]
            {row['answer']} </s>
            """
        return input_prompt
    else:
        addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
        input_prompt =f"""<s> [INST] {addon_prompt}\n
            {row['json_encoding']}\n
            {row['question']}\n
            [/INST]
            """
        return input_prompt

def create_translation_prompt(row, mode='train'):
    # print(prompt_translation(row, mode)[0])
    return prompt_translation(row, mode)

def generate_data(dpath):
    translation_ds = load_dataset("json", data_files=dpath)
    print(translation_ds)
    train_dataset = translation_ds["train"].select([i for i in range(15000)])
    eval_dataset = translation_ds["train"].select([i for i in range(15000, 17594)])
    return train_dataset, eval_dataset


def lora_setup():
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
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
        )

    return model_kwargs, peft_config

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
                   
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
# trainer = CustomSFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     eval_dataset=val_dataset,
#     peft_config=peft_config,
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
#     packing=True,
#     formatting_func=format_instruction,
#     args=args,
# )
# trainer.train()

def train(train_dataset, eval_dataset, OUTPUT_PATH, args):
    output_dir = OUTPUT_PATH
    total_num_steps = args.num_train_epochs * len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)

    model_kwargs, peft_config = lora_setup()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size//2,
        bf16=True,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.1,
        max_steps=total_num_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        evaluation_strategy="steps",
        eval_steps=100,
        # logging strategies
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
    )

    trainer = CustomSFTTrainer(
        model=MODEL_ID,
        model_init_kwargs=model_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        max_seq_length=1024,
        args=training_args,
        formatting_func=create_translation_prompt,
        peft_config=peft_config,
    )

    trainer.train()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning Mistral OpenOrca.')
    parser.add_argument('--dataset_format', default="json", help='Format of dataset for training')
    # parser.add_argument('dataset_path', default="../data/json_encoded_convfinqa.json", help='Path to training file')
    parser.add_argument('--batch_size', default=6, help='Batch Size',  type=int)
    parser.add_argument('--num_train_epochs', default=3, help='Batch Size', type = int)
    parser.add_argument('--gradient_accumulation_steps', default=32, help='Gradient Accumulation Steps', type=int)
    # parser.add_argument('output', default=f"output/{args.dataset_format}_encodings", help='Output Path')
    # 3 * (6 * 32)

    args = parser.parse_args()

    DATASET_PATH = f"../data/{args.dataset_format}_encoded_convfinqa.json"
    OUTPUT_PATH = f"output/{args.dataset_format}_encodings"
    # train_dataset, eval_dataset = generate_data(args.dataset_path)
    train_dataset, eval_dataset = generate_data(DATASET_PATH)

    train(train_dataset, eval_dataset, OUTPUT_PATH, args)