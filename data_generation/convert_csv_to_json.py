import csv
import json

csv_file_path = '../data/convfinqa_json_max_length_2048.csv'
json_file_path = '../data/convfinqa_json_max_length_2048.json'

data = []

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)

print(f"Conversion complete. JSON file saved at: {json_file_path}")
