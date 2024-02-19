from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig
from tqdm import tqdm
import pandas as pd
import torch
import re

# model_id = "Open-Orca/Mistral-7B-OpenOrca"
model_id ="llmware/dragon-mistral-7b-v0"
# adapter_model_id = "/home/ubuntu/llm/Large-Language-Model-for-Key-Value-Extractions/notebooks/output/pipe_encodings/checkpoint-125"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# peft_config = PeftConfig.from_pretrained(adapter_model_id)

# # to initiate with random weights
# peft_config.init_lora_weights = False

# model.add_adapter(peft_config)
# model.enable_adapters()

df = pd.read_csv('../data/json_encoded_convfinqa.csv')

example_context = """<html>
<body>
  "three factor formula" ) . the consolidated financial statements include northrop grumman management and support services allocations totaling $ 32 million for the year ended december 31 , 2011 . shared services and infrastructure costs - this category includes costs for functions such as information technology support , systems maintenance , telecommunications , procurement and other shared services while hii was a subsidiary of northrop grumman . these costs were generally allocated to the company using the three factor formula or based on usage . the consolidated financial statements reflect shared services and infrastructure costs allocations totaling $ 80 million for the year ended december 31 , 2011 . northrop grumman-provided benefits - this category includes costs for group medical , dental and vision insurance , 401 ( k ) savings plan , pension and postretirement benefits , incentive compensation and other benefits . these costs were generally allocated to the company based on specific identification of the benefits provided to company employees participating in these benefit plans . the consolidated financial statements include northrop grumman- provided benefits allocations totaling $ 169 million for the year ended december 31 , 2011 . management believes that the methods of allocating these costs are reasonable , consistent with past practices , and in conformity with cost allocation requirements of cas or the far . related party sales and cost of sales prior to the spin-off , hii purchased and sold certain products and services from and to other northrop grumman entities . purchases of products and services from these affiliated entities , which were recorded at cost , were $ 44 million for the year ended december 31 , 2011 . sales of products and services to these entities were $ 1 million for the year ended december 31 , 2011 . former parent's equity in unit transactions between hii and northrop grumman prior to the spin-off have been included in the consolidated financial statements and were effectively settled for cash at the time the transaction was recorded . the net effect of the settlement of these transactions is reflected as former parent's equity in unit in the consolidated statement of changes in equity . 21 . unaudited selected quarterly data unaudited quarterly financial results for the years ended december 31 , 2013 and 2012 , are set forth in the following tables: .
  {'sales and service revenues': {'year ended december 31 2013 1st qtr': '$ 1562', 'year ended december 31 2013 2nd qtr': '$ 1683', 'year ended december 31 2013 3rd qtr': '$ 1637', 'year ended december 31 2013 4th qtr': '$ 1938'}, 'operating income ( loss )': {'year ended december 31 2013 1st qtr': '95', 'year ended december 31 2013 2nd qtr': '116', 'year ended december 31 2013 3rd qtr': '127', 'year ended december 31 2013 4th qtr': '174'}, 'earnings ( loss ) before income taxes': {'year ended december 31 2013 1st qtr': '65', 'year ended december 31 2013 2nd qtr': '87', 'year ended december 31 2013 3rd qtr': '99', 'year ended december 31 2013 4th qtr': '143'}, 'net earnings ( loss )': {'year ended december 31 2013 1st qtr': '44', 'year ended december 31 2013 2nd qtr': '57', 'year ended december 31 2013 3rd qtr': '69', 'year ended december 31 2013 4th qtr': '91'}, 'dividends declared per share': {'year ended december 31 2013 1st qtr': '$ 0.10', 'year ended december 31 2013 2nd qtr': '$ 0.10', 'year ended december 31 2013 3rd qtr': '$ 0.10', 'year ended december 31 2013 4th qtr': '$ 0.20'}, 'basic earnings ( loss ) per share': {'year ended december 31 2013 1st qtr': '$ 0.88', 'year ended december 31 2013 2nd qtr': '$ 1.14', 'year ended december 31 2013 3rd qtr': '$ 1.38', 'year ended december 31 2013 4th qtr': '$ 1.86'}, 'diluted earnings ( loss ) per share': {'year ended december 31 2013 1st qtr': '$ 0.87', 'year ended december 31 2013 2nd qtr': '$ 1.12', 'year ended december 31 2013 3rd qtr': '$ 1.36', 'year ended december 31 2013 4th qtr': '$ 1.82'}}
  .
 </body>
</html>
"""

example_question ="""What is income from discontinued operations before income taxes for year ended december 31 2009 ( in thousands ) ?"""

example_answer = """$5,367"""
def prompt_translation(row):
    eval_mode = True
    # addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
    addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully. Give exact answers only."

    input_prompt =f"""<s> [INST] {addon_prompt}\n
        {row['json_encoding']}\n
        {row['question']}\n
        [/INST]
        """

    # input_prompt =f"""<s> [INST] {addon_prompt}\n
    #     {example_context}\n
    #     {example_question}\n
    #     [/INST]
    #     {example_answer}
    #     </s><s>
    #     [INST] {addon_prompt}\n
    #     {row['json_encoding']}\n
    #     {row['question']}\n
    #     [/INST]
    #     """
    # input_prompt =f"""<|im_start|>system
    #         {addon_prompt}
    #         <|im_end|>
    #         <|im_start|>user
    #         {example_context}
    #         {example_question}
    #         <|im_start|>assistant
    #         {example_answer}<|im_end|>
    #         |im_start|>system
    #         {addon_prompt}
    #         <|im_end|>
    #         <|im_start|>user
    #         {row['json_encoding']}
    #         {row['question']}<|im_end|>
    #         <|im_start|>assistant
    #     """
    return input_prompt

# def prompt_translation(row):
#     new_prompt = "<human>: " + row['json_encoding'] + "\n" + row['question'] + "\n" + "<bot>:"
#     return new_prompt

output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction'])

for i, row in tqdm(df[:100].iterrows()):
    text = prompt_translation(row)
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    # inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    # output = model.generate(**inputs)
    # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    decoded_output = tokenizer.decode(model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
    # Extract values after [/INST]
    inst_values = decoded_output.split('[/INST]')[1].strip()
    # Define the regular expression pattern
    # pattern = re.compile(r'\[/INST\].*?\[/INST\](.*)')

    # # Search for the pattern in the input prompt
    # match = pattern.search(decoded_output)

    # inst_values = match.group(1).strip()
    # parts = decoded_output.split("assistant")

    # # Extract all values after the last occurrence of "assistant"
    # inst_values = [part.strip() for part in parts[-1].split("\n") if part.strip()][-1]

    print(inst_values)
    print('***************')
    output_df.at[i, 'index'] = row['index']
    output_df.at[i, 'question'] = row['question']
    output_df.at[i, 'answer'] = row['answer']
    output_df.at[i, 'prediction'] = inst_values

# output_df.to_csv('../output/test_mistral_dragom_model.csv', index=False)
output_df.to_csv('../output/test_exact_answer_instruction.csv', index=False)

