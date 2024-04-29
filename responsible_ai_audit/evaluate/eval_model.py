#OK, all of the code below is just to set up a "black box" model to code with
from typing import Union
import torch.amp
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax
from datasets import load_dataset
import torch
import seaborn as sns
from responsible_ai_audit.data import get_data
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn

#sns formatting
sns.set_context("poster")
sns.set_theme(style="whitegrid", palette="pastel")

#functions
def get_max_label(softmax_scores):
    max_labels = []
    for i in range(0,len(softmax_scores)):
        scores = softmax_scores[i]
        max_score = max(scores)
        if(max_score==scores[0]):
            max_labels.append(0)
        elif(max_score==scores[1]):
            max_labels.append(1)
        else:
            max_labels.append(2)
    return max_labels

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def get_val_dataset():
    dataset = load_dataset("social_bias_frames", split = "test")[:1000]
    return dataset

def get_model(model_name: str, num_labels: int = 3, base_dir = Path("models"), device: Union[str, torch.device] = "cpu") -> nn.Module:
    base_model = "distilbert/distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, device_map=device)
    state_dict = torch.load(base_dir.joinpath(model_name+".pth"), map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_df_w_predictions(val_data, model, tokenizer):
    dataset_text = val_data["post"]
    dataset_race = val_data["annotatorRace"]
    dataset_gender = val_data["annotatorGender"]
    dataset_politics = val_data["annotatorPolitics"]
    dataset_offensiveYN = val_data["offensiveYN"]
    #preprocess and encode
    processed_texts = [preprocess(text) for text in dataset_text]
    encoded_inputs = [tokenizer(processed_text, return_tensors="pt") for processed_text in processed_texts]
    print("done with preprocess and encode")
    #get model outputs
    outputs = []
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = [model(**encoded_input) for encoded_input in encoded_inputs]
    scores = [output[0][0].detach().float().numpy() for output in outputs]
    softmax_scores = [softmax(score) for score in scores]
    #
    offensives = [softmax_score[0] for softmax_score in softmax_scores]
    neutrals = [softmax_score[1] for softmax_score in softmax_scores]
    not_offensives = [softmax_score[2] for softmax_score in softmax_scores]
    print("done with positive/negative/neutral")
    #Build a dataframe
    df = pd.DataFrame({'Posts':dataset_text, 'Annotator Race':dataset_race,'Annotator Gender':dataset_gender,'Annotator Politics':dataset_politics,'offensiveYN':dataset_offensiveYN, 'OffensiveY Score': offensives, 'Neutral Score':neutrals, 'OffensiveN Score':not_offensives, 'Prediction': get_max_label(softmax_scores)})
    return df


def get_dfs_with_labels(val_dataset):
    name_list = ["white_male_conservative", "white_male_liberal", "white_female_liberal", "black_female_moderate_liberal", "white_female_conservative", "all"]
    url = "distilbert/distilroberta-base"
    dfs_with_labels = []
    for name in name_list:
        tokenizer = AutoTokenizer.from_pretrained(url)
        model = get_model(name, base_dir=Path("/Users/samuelwu/Desktop/Senior/Spring/DS682/Final Project/responsible-ai-audit/responsible_ai_audit/evaluate/models"))
        dfs_with_labels.append((get_df_w_predictions(val_dataset,model,tokenizer), name))
    return dfs_with_labels

#bar plot
def bar_plot(df_with_label):
    df = df_with_label[0]
    label = df_with_label[1]
    fig = df['Prediction'].value_counts().plot(kind='bar')
    plt.title("Prediction Distribution: "+label, fontsize=16)
    plt.show()
    #fig.savefig('assets/barplot.png')

#plot TP/TN/FP/FN 

def FPR_bar_plot(dfs_with_labels):
    label = []
    value = []
    model = []
    for df,model_name in dfs_with_labels:
        model = model+ [model_name, model_name, model_name, model_name]
        label = label + [model_name+" TP", model_name+" TN", model_name+" FP", model_name+" FN"]

        value = value+[sum(offensiveYN == "1.0" and prediction == 0 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == "0.0" and prediction == 2 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == "0.0" and prediction == 0 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == "1.0" and prediction == 2 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"]))]
    FPR_df = pd.DataFrame({"Label":label, "Value":value, "Model":model})
    sns.set_theme(rc={'figure.figsize':(80,8.27)})
    plt.title("TP/TN/FP/FN Distribution", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("# of Predictions", fontsize=14)

    # Display the plot
    sns.barplot(x = 'Label',y = 'Value',hue = 'Model',data = FPR_df)
    plt.show()

def ROC_plot(df_with_label):
    df = df_with_label[0]
    label = df_with_label[1]
    offYN = [1 if elem == "1.0" else 0 for elem in df["offensiveYN"]]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(offYN, df["OffensiveY Score"])
    roc_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
    sns.set_theme(rc={'figure.figsize':(11,8)})
    plt.title("ROC Curve: "+label, fontsize=16)
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)

    # Display the plot
    sns.lineplot(data = roc_df, x = "fpr", y = "tpr")
    plt.show()
    


dfs = get_dfs_with_labels(get_val_dataset())

FPR_bar_plot(dfs)
ROC_plot(dfs[0])
bar_plot(dfs[0])

