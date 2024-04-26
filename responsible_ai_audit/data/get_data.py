from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

dataset = load_dataset("social_bias_frames")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_and_tokenize(post):
  encoded_input = tokenizer(post, return_tensors='pt')
  return encoded_input

def unpack_encoded_input(row):
    input_ids, attention_mask = row
    return pd.Series({'input_ids': input_ids, 'attention_mask': attention_mask})

def get_train_val(dataset):
    train = dataset["train"]
    validation = dataset["validation"]
    train_df = train.to_pandas()
    val_df = validation.to_pandas()

    # merge datasets and group by post to get majority label
    merged = pd.concat([train_df, val_df])
    grouped = merged.groupby('post')['offensiveYN'].agg(lambda x: x.value_counts().idxmax()).reset_index()

    # tokenize post column, convert encoded input to 2 columns to avoid issues when converting to torch dataset
    grouped['encoded_input'] = grouped['post'].apply(preprocess_and_tokenize)
    grouped[['input_ids', 'attention_mask']] = grouped['encoded_input'].apply(unpack_encoded_input)
    grouped = grouped.drop(columns=['encoded_input'])
    
    # split into 80/20 train/val split
    train_split, val_split = train_test_split(grouped, test_size=0.2, random_state=88)

    # convert to torch dataset
    train_dataset = Dataset.from_pandas(train_split)
    val_dataset = Dataset.from_pandas(val_split)
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    return train_dataset, val_dataset

# 4698 rows
def get_test():
    test = dataset["test"]
    test_df = test.to_pandas()
    grouped = test_df.groupby('post')['offensiveYN'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    grouped['encoded_input'] = grouped['post'].apply(preprocess_and_tokenize)
    grouped[['input_ids', 'attention_mask']] = grouped['encoded_input'].apply(unpack_encoded_input)
    grouped = grouped.drop(columns=['encoded_input'])

    test_dataset = Dataset.from_pandas(grouped)
    test_dataset.set_format("torch")
    return test_dataset

# other populations to explore - hisp man: 3k - liberal: 900, cons-1.8k, hisp woman: 4k - mod-liberal: 1k, other: 2k

# white male conservative - train/val = 5184/1296
def getWhiteMaleConsData():
    white_male_cons = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                        example['annotatorGender'] == 'man' and
                                                        example['annotatorPolitics'] == 'cons')
    return get_train_val(white_male_cons)

# white male liberal - train/val = 10,404/2601   
def getWhiteMaleLibData():
    white_male_liberal = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                        example['annotatorGender'] == 'man' and
                                                        example['annotatorPolitics'] == 'liberal')
    return get_train_val(white_male_liberal)

# white female liberal - train/val = 14,557/3640
def getWhiteFemaleLibData():
    white_female_liberal = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                      example['annotatorGender'] == 'woman' and
                                                      example['annotatorPolitics'] == 'liberal')
    return get_train_val(white_female_liberal)

# white female conservative - train/val = 756/190
def getWhiteFemaleConsData():
    white_female_cons = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                    example['annotatorGender'] == 'woman' and
                                                    example['annotatorPolitics'] == 'cons')
    return get_train_val(white_female_cons)

# black female mod-liberal - train/val = 3025/757
def getBlackFemaleModlibData():
    black_female_modliberal = dataset.filter(lambda example: example['annotatorRace'] == 'black' and
                                                            example['annotatorGender'] == 'woman' and
                                                            example['annotatorPolitics'] == 'mod-liberal')
    return get_train_val(black_female_modliberal)

###
# sample use:
# train, val = getWhiteMaleConsData()
# print(len(train))
# print(len(val))
###
