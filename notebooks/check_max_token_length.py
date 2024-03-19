from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from collections import Counter
import csv

model_id ="llmware/dragon-mistral-7b-v0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

df = pd.read_csv('../data/pseudo_json_dataset_V2.csv')

print(df.shape)
addon_prompt = """Read the following texts and table with financial data from an S&P 500 earnings report carefully. 
Provide the exact answer only. If there are multiple answers for the questions, explain each answers."""

token_len = [] 
for index, row in tqdm(df.iterrows()):
    input_prompt = f"<human>: {addon_prompt}\n {row['pseudo_json_encoding']}\n {row['question']}\n <bot>: {row['answer']}"
    input_prompt.replace(' ', '')
    inputs = tokenizer(input_prompt, return_tensors="pt").to('cuda')
    length = len(inputs['input_ids'][0])
    token_len.append(length)
    if length > 2048:
        df.drop(index, inplace=True)
count_dict = Counter(token_len)
sorted_values = sorted(count_dict.items(), key=lambda x: x[0], reverse=True)
# sorted_list = sorted(set(token_len), reverse=True)       
# print(sorted_list[:10])  
# sorted_list = sorted(set(token_len))       
# print(sorted_list[:10])  
df.to_csv('../data/pseudo_json_dataset_V2_2048.csv', index=False)
print(df.shape)
with open("../data/token_len_sorted_values.csv", mode='w', newline='')  as file:
    writer = csv.writer(file)
    writer.writerow(['value', 'count'])
    for value, count in sorted_values:
        writer.writerow([value, count])                