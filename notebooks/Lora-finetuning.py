from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from datetime import datetime
import transformers
import pandas as pd
import torch


csv_file = pd.read_csv('../data/index_info.csv')
index_info = csv_file['Index of data without headers'].tolist()

translation_ds = load_dataset("json", data_files='../data/json_encoded_convfinqa.json')

train_dataset = translation_ds["train"].select([i for i in range(15000)])
eval_dataset = translation_ds["train"].select([i for i in range(15000, 17594)])

def format_dataset(row):
    addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
    input_prompt =f"""<s> [INST] {addon_prompt}\n
        {row['json_encoding']}\n
        {row['question']}\n
        [/INST]
        {row['answer']} </s>
        """
    return {"text": input_prompt}


base_model_id = "Open-Orca/Mistral-7B-OpenOrca"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2048,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.unk_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(row):
    addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
    full_prompt =f"""<s> [INST] {addon_prompt}\n
        {row['json_encoding']}\n
        {row['question']}\n
        [/INST]
        {row['answer']} </s>
        """
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)
# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)


project = "json_format"
base_model_name = "quantized_mistral_model"
run_name = base_model_name + "-" + project
output_dir = "../output/" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=16,
        max_steps=1000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()