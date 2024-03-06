from numpy import dsplit
from numpy import dsplit
import wandb
from pathlib import Path
import torch
from transformers import TrainingArguments
import datasets 
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import sys
import os
sys.path.append(Path(os.getcwd()).parent)
# from llm_recipes.utils import LLMSampleCB


wandb.login()


def prompt_formatter(row, mode='train'):
    if mode == 'train':
        return (
            f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
            This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
            The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


            ### Target sentence:
            {row["target"]}


            ### Meaning representation:
            {row["meaning_representation"]}
            """
            )
    else:
        row["meaning_representation"] = ''
        return (
            f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
            This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
            The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


            ### Target sentence:
            {row["target"]}


            ### Meaning representation:
            {row["meaning_representation"]}
            """
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
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
    # target_modules = ["model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.k_proj", "model.layers.31.self_attn.v_proj",
    #                   "model.layers.31.self_attn.o_proj", "model.layers.30.self_attn.q_proj", "model.layers.30.self_attn.k_proj", 
    #                   "model.layers.30.self_attn.v_proj", "model.layers.30.self_attn.o_proj",]
)


batch_size = 1
# 3 * (4 * 32)
num_train_epochs = 2
gradient_accumulation_steps = 128
total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)

output_dir = "results_V5/"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    bf16=True,
    learning_rate=0.0003,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
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
    load_best_model_at_end=True,
)

trainer = SFTTrainer(
    model=model_id,
    model_init_kwargs=model_kwargs,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset,
    packing=True,
    max_seq_length=2048,
    args=training_args,
    formatting_func=create_formatted_prompt,
    peft_config=peft_config,
)


# remove answers
def create_prompt_no_anwer(row):
    return {"text": create_formatted_prompt(row, mode='eval')}


test_dataset = eval_dataset.map(create_prompt_no_anwer)
wandb_callback = LLMSampleCB(trainer, eval_dataset, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)


trainer.train()
wandb.finish()
