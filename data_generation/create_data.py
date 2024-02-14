import pandas as pd
from bs4 import BeautifulSoup
import random
import re
from tqdm import tqdm

csv_file = pd.read_csv('index_info.csv')
index_info = csv_file['Index of data without headers'].tolist()

def beautify_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    return table

final_df = pd.DataFrame(columns = ['index', 'html_content', 'question', 'answer'])
df = pd.read_csv('dataset_index.csv')
count = 0
for _, row in tqdm(df.iterrows()):
    table = beautify_html(row['html_content'])

    for i in range(50):
        # print('here is index', type(row['index']))
        if int(row['index']) in index_info:
            first_col = [row.find_all('td')[1].text for row in table.find_all('tr')]
            random_index = random.choice(range(len(first_col)))
            second_col = [row.find_all('td')[2].text for row in table.find_all('tr')]
            system_prompt = f"What is {first_col[random_index]}?"
            system_prompt = system_prompt.replace('\n', '')
            answer = second_col[random_index]
        else:
            first_row = [td.text for td in table.find('tr').find_all('td')]
            # print('first row', first_row, len(first_row))

            random_row_index = random.randint(2, len(first_row) - 1)

            first_col = [row.find_all('td')[1].text for row in table.find_all('tr')[1:]]
            # print('first col', first_col, len(first_col))

            if len(first_col) == 1:
                random_col_index = 0
            else:
                random_col_index = random.choice(range(len(first_col) - 1))

            # print('row random', random_row_index)
            # print('col random', random_col_index)

            random_col_values = [row.find_all('td')[random_row_index].text for row in table.find_all('tr')[1:]]
            # print('col values', random_col_values, len(random_col_values))
            # print(first_col[random_col_index])
            col_data = first_col[random_col_index].replace('\n', '')
            row_data = first_row[random_row_index].replace('\n', '')
            system_prompt = f"What is {col_data} for {row_data}?"
            # system_prompt = f"What is {first_col[random_col_index]} for {first_row[random_row_index]}?"

            # print(system_prompt)
            # print(random_col_values[random_col_index].replace('\n', ''))
            # print(random_col_values[random_col_index])
            answer = random_col_values[random_col_index].replace('\n', '')
        system_prompt = re.sub(r'\s+', ' ', system_prompt)
        answer = re.sub(r'\s+', ' ', answer)
        print(system_prompt)
        print(answer)
        print('****************')
        final_df.loc[len(final_df)] = {"index": row['index'], 'html_content': row['html_content'] ,'question': system_prompt, 'answer': answer}
        final_df = final_df.drop_duplicates(subset=['index', 'question', 'answer'])
    count+=1
    if count==4:
        break
# import pdb; pdb.set_trace()
# Remove duplicates based on two columns (Column1 and Column2)
final_df_no_duplicates = final_df.drop_duplicates(subset=['question', 'answer'])
final_df_no_duplicates.to_csv('datasettest1.csv', index = False)