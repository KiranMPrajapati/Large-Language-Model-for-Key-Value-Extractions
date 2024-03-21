from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import torch
import re
from bs4 import BeautifulSoup
from io import StringIO
import csv 

df = pd.read_csv('../data/pseudo_json_dataset_V3_2048.csv')
table_descriptions = "The csv contains financial data from an S&P 500 earnings."

def convert_to_csv_in_stringio_format(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table
    table = soup.find('table')
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

def convert_to_csv(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table
    table = soup.find('table')
    # Open StringIO object for writing
    csv_buffer = StringIO()


    # Write rows to the StringIO object
    writer = csv.writer(csv_buffer)
    trows = table.find_all('tr')
    final_csv = ''
    for trow in trows:
        writer = csv.writer(csv_buffer)
        writer.writerow([cell.get_text() for cell in trow.find_all(['td', 'th'])])
        csv_data = csv_buffer.getvalue().replace('\n', '').replace('  ', '')
        csv_buffer = StringIO()
        final_csv += csv_data + '\n'
    return final_csv

df['csv_content'] = df['html_content'].apply(convert_to_csv)
def create_prompt(row):
    csv_content = row['csv_content']
    question = row['question']
    return (
        f"""[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].
        ### [Table Text]
        {table_descriptions}

        ### [Table]
        ```
        {csv_content}
        ```

        ### [Question]
        {question}

        ### [Solution][/INST]"""
    )

df["value"] = df.apply(create_prompt, axis=1)
df = df.dropna()
print(df.columns)
custom_ds = pd.DataFrame()
custom_ds["prompt"] = df['value']

train_dataset = Dataset.from_pandas(custom_ds)

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "/home/ubuntu/Large-Language-Model-for-Key-Value-Extractions/notebooks/results_V12/checkpoint-160"
config = PeftConfig.from_pretrained(peft_model_id)


# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  device_map={"":0}, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

def extract_inst(output):
    # Find the index of [/INST]
    index = output.find("[/INST]")

    # Extract the value after [/INST]
    value_after_inst = output[index + len("[/INST]"):].strip()
    return value_after_inst 

output_df = pd.DataFrame(columns=['index', 'csv_content', 'question', 'answer', 'prediction'])

for i in tqdm(range(0, 2000)):
    try:
        text = train_dataset["prompt"][i]
        input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids.cuda()
        # with torch.inference_mode():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=30, do_sample=True, top_p=0.9)
        decoded_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        
        inst_values = extract_inst(decoded_output)
        # print(decoded_output)
        # print(inst_values)
        # # print(df.loc[i,'csv_content'].replace('  ', ''))
        # print(df.loc[i,'question'])
        # print(df.loc[i,'answer'])
        print('***************')
        output_df.at[i, 'index'] = df.loc[i, 'index']
        output_df.at[i, 'question'] = df.loc[i,'question']
        output_df.at[i, 'answer'] = df.loc[i,'answer']
        output_df.at[i, 'csv_content'] = df.loc[i,'csv_content'].replace('  ', '')
        output_df.at[i, 'prediction'] = inst_values
    except:
        print(f'error')

output_df.to_csv('results_V12/train_sample.csv', index=False)

