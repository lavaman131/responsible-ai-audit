#OK FUCKERS LETS DO THIS SHIT
#God i just want to die
#OK, all of the code below is just to set up a "black box" model to code with
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax
from datasets import load_dataset

#functions
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

def get_max_label(softmax_scores):
    max_labels = []
    for i in range(0,len(softmax_scores)):
        scores = softmax_scores[i]
        max_score = max(scores)
        if(max_score==scores[0]):
            max_labels.append("Negative")
        elif(max_score==scores[1]):
            max_labels.append("Neutral")
        else:
            max_labels.append("Positive")
    return max_labels

#model setup
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#ok cool, now we set up the data_set. in the actual we can import data/thingy.py and load from there
#but for now we'll just make up some fake dataset
from responsible_ai_audit.data import get_data
dataset = load_dataset("social_bias_frames")

training_data, val_data = get_data.get_train_val_split(dataset, get_data.getWhiteMaleLibData)
dataset_text = training_data["post"]

processed_texts = [preprocess(text) for text in dataset_text]
encoded_inputs = [tokenizer(processed_text, return_tensors="pt") for processed_text in processed_texts]

#This is my "black box" output of our model
outputs = [model(**encoded_input) for encoded_input in encoded_inputs]
scores = [output[0][0].detach().numpy() for output in outputs]
softmax_scores = [softmax(score) for score in scores]

negatives = [softmax_score[0] for softmax_score in softmax_scores]
neutrals = [softmax_score[1] for softmax_score in softmax_scores]
positives = [softmax_score[2] for softmax_score in softmax_scores]

#Build a dataframe
df = pd.DataFrame({'Posts':dataset_text, 'Negative Score':negatives, 'Neutral Score':neutrals, 'Positive Score':positives, 'Label': get_max_label(softmax_scores)})

#bar plot
df['Label'].value_counts().plot(kind='bar')

#plot TP/TN/FP/FN
valid_values = val_data["offensiveYN"]
print(valid_values)

#
