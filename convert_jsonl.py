import pandas as pd
import json

def convert_jsonl_to_csv(input_file, output_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

convert_jsonl_to_csv('data/1english_openpii_30k.jsonl', 'data/ai4privacydata1.csv')
convert_jsonl_to_csv('data/1english_openpii_8k.jsonl', 'data/ai4privacydata2.csv')

# load and concatenate the two datasets
df1 = pd.read_csv('data/ai4privacydata1.csv')
df2 = pd.read_csv('data/ai4privacydata2.csv')
df = pd.concat([df1, df2], ignore_index=True)
df.to_csv('data/ai4privacydata.csv', index=False)