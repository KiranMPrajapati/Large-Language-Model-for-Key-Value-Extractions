from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import torch
import re

df = pd.read_csv('../data/convfinqa_json_max_length_2048.csv')
df["value"] = (
    "<human>:" +
    df['json_encoding'] + "\n" +
    df['question'] + "\n" +
    "<bot>:"
)
df = df.dropna()
print(df.columns)
custom_ds = pd.DataFrame()
custom_ds["prompt"] = df['value']

train_dataset = Dataset.from_pandas(custom_ds)

# model_id = "Open-Orca/Mistral-7B-OpenOrca"
model_id ="llmware/dragon-mistral-7b-v0"
adapter_model_id = "/home/ubuntu/llm/Large-Language-Model-for-Key-Value-Extractions/notebooks/results/checkpoint-460"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

lora_config = LoraConfig.from_pretrained(adapter_model_id)
model = get_peft_model(model, lora_config)


print("Test sample")
text = train_dataset["prompt"][10]
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def extract_bot(output):
    # Splitting the conversation text by lines
    lines = output.split('\n')

    # Extracting the response after <bot>:
    bot_response = None
    for line in lines:
        if line.strip().startswith("<bot>:"):
            bot_response = line.strip().split("<bot>:")[1].strip()
            break
    return bot_response
output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction'])

for i in tqdm(range(20, 13000)):
    try:
        inputs = tokenizer(train_dataset["prompt"][i], return_tensors="pt").to('cuda')
        # inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        # output = model.generate(**inputs)
        # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        decoded_output = tokenizer.decode(model.generate(input_ids=inputs["input_ids"],  attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
        inst_values = extract_bot(decoded_output)
        print(inst_values)
        print(df.loc[i,'question'])
        print('***************')
        output_df.at[i, 'index'] = df.loc[i, 'index']
        output_df.at[i, 'question'] = df.loc[i,'question']
        output_df.at[i, 'answer'] = df.loc[i,'answer']
        output_df.at[i, 'prediction'] = inst_values
    except:
        print(f'error')

# output_df.to_csv('../output/test_mistral_dragom_model.csv', index=False)
output_df.to_csv('results/results_train_v3.csv', index=False)

