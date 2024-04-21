#OK FUCKERS LETS DO THIS SHIT
#God i just want to die
#OK, all of the code below is just to set up a "black box" model to code with
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#ok cool, now we set up the data_set. in the actual we can import data/thingy.py and load from there
#but for now we'll just make up some fake dataset
import load_data
data = load_data.getRandDataset(10)

dataset_text = [""]
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors="pt")

#This is my "black box" output of our model
output = model(**encoded_input)
scores = output[0][0].detach().numpy()

#These are the scores I'll be analyzing 
scores = softmax(scores)

print(scores)