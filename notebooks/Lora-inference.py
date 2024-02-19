from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd
import torch 

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

# eval_prompt = """<s> [INST] Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.
# col : 1 | ( $ in millions except per share amounts ) | year ended december 31 2013 1st qtr | year ended december 31 2013 2nd qtr | year ended december 31 2013 3rd qtr | year ended december 31 2013 4th qtr
# row 1 : 2 | sales and service revenues | $ 1562 | $ 1683 | $ 1637 | $ 1938
# row 2 : 3 | operating income ( loss ) | 95 | 116 | 127 | 174
# row 3 : 4 | earnings ( loss ) before income taxes | 65 | 87 | 99 | 143
# row 4 : 5 | net earnings ( loss ) | 44 | 57 | 69 | 91
# row 5 : 6 | dividends declared per share | $ 0.10 | $ 0.10 | $ 0.10 | $ 0.20
# row 6 : 7 | basic earnings ( loss ) per share | $ 0.88 | $ 1.14 | $ 1.38 | $ 1.86
# row 7 : 8 | diluted earnings ( loss ) per share | $ 0.87 | $ 1.12 | $ 1.36 | $ 1.82
# What is sales and service revenues for year ended december 31 2013 2nd qtr ?
# [/INST]
# """
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
# print(output)

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


df = pd.read_csv('../data/json_encoded_convfinqa.csv')

def format(row):
    eval_mode = True
    addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully. Generate exact answer to the point."
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
    input_prompt =f"""<s> [INST] {addon_prompt}\n
        {row['json_encoding']}\n
        {row['question']}\n
        [/INST]
        """
    # input_prompt =f"""<|im_start|>system
    #         {addon_prompt}
    #         <|im_end|>
    #         <|im_start|>user
    #         {example_context}
    #         {example_question}
    #         <|im_start|>assistant
    #         {example_answer}<|im_end|>
            # |im_start|>system
            # {addon_prompt}
            # <|im_end|>
    #         <|im_start|>user
    #         {row['json_encoding']}
    #         {row['question']}<|im_end|>
    #         <|im_start|>assistant
    #     """

    input_prompt =f"""|im_start|>system
            {addon_prompt}
            <|im_end|>
            <|im_start|>user
            {row['json_encoding']}
            {row['question']}<|im_end|>
            <|im_start|>assistant
        """
    return input_prompt

output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction'])
import re
model.eval()
with torch.no_grad():
    for i, row in tqdm(df[:10].iterrows()):
        eval_prompt = format(row)
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
        inst_values = output.split('[/INST]')[1].strip()
        # print(output)

        # pattern = re.compile(r'\[/INST\](.*?)</s>', re.DOTALL)

        # # Search for the pattern in the input prompt
        # match = pattern.findall(output)

        # inst_values = match[-1].strip()

        # parts = output.split("assistant")

        # # Extract all values after the last occurrence of "assistant"
        # inst_values = [part.strip() for part in parts[-1].split("\n") if part.strip()]

        print(inst_values)
        output_df.at[i, 'index'] = row['index']
        output_df.at[i, 'question'] = row['question']
        output_df.at[i, 'answer'] = row['answer']
        output_df.at[i, 'prediction'] = inst_values
output_df.to_csv('../output/json_zero_shot_4_bit_exact_answer.csv', index=False)

# df = pd.read_csv('../data/pipe_encoded_convfinqa.csv')

# def format_pipe(row):
#     eval_mode = True
#     addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
#     input_prompt =f"""<s> [INST] {addon_prompt}\n
#         {row['PIPE_encoding']}\n
#         {row['question']}\n
#         [/INST]
#         """
#     return input_prompt

# output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction'])

# model.eval()
# with torch.no_grad():
#     for i, row in tqdm(df[:100].iterrows()):
#         eval_prompt = format_pipe(row)
#         model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#         output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
#         inst_values = output.split('[/INST]')[1].strip()
#         output_df.at[i, 'index'] = row['index']
#         output_df.at[i, 'question'] = row['question']
#         output_df.at[i, 'answer'] = row['answer']
#         output_df.at[i, 'prediction'] = inst_values
# output_df.to_csv('../output/pipe_zero_shot_4_bit.csv', index=False)

# df = pd.read_csv('../data/dataset_with_all_encodings.csv')

# def format_html(row):
#     eval_mode = True
#     addon_prompt = "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided."
#     input_prompt =f"""<s> [INST] {addon_prompt}\n
#         {row['html_content']}\n
#         {row['question']}\n
#         [/INST]
#         """
#     return input_prompt

# output_df = pd.DataFrame(columns=['index', 'question', 'answer', 'prediction'])

# model.eval()
# with torch.no_grad():
#     for i, row in tqdm(df[:100].iterrows()):
#         eval_prompt = format_html(row)
#         model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#         output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
#         inst_values = output.split('[/INST]')[1].strip()
#         output_df.at[i, 'index'] = row['index']
#         output_df.at[i, 'question'] = row['question']
#         output_df.at[i, 'answer'] = row['answer']
#         output_df.at[i, 'prediction'] = inst_values
# output_df.to_csv('../output/html_zero_shot_4_bit.csv', index=False)