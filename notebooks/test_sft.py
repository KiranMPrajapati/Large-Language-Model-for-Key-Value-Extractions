import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import warnings
import wandb

warnings.filterwarnings("ignore")

wandb.login()

# def format_prompt(row):
#     input_prompt = f"<human>: {row['json_encoding']}\n {row['question']}\n <bot>: {row['answer']}"
#     return {'prompt': input_prompt}

df = pd.read_csv('../data/json_encoded_convfinqa.csv')
df["value"] = (
    "<human>:" +
    df['json_encoding'] + "\n" +
    df['question'] + "\n" +
    "<bot>:" +
    df['answer']
)
df = df.dropna()

custom_ds = pd.DataFrame()
custom_ds["prompt"] = df['value']

dataset = Dataset.from_pandas(custom_ds)
model_name ="llmware/dragon-mistral-7b-v0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, quantization_config=bnb_config, trust_remote_code=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True
)

model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

output_dir = "../results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 32
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "cosine"
num_train_epochs = 3

max_steps = num_train_epochs * len(dataset) // (per_device_train_batch_size * gradient_accumulation_steps)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    # optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    # warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()
wandb.finish()
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("../outputs")
