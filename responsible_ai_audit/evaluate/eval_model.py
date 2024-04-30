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
sns.set_theme(style="whitegrid", palette="pastel", font_scale=2, rc={'figure.figsize':(11,11)}, font ='Times New Roman')

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
    dataset = load_dataset("social_bias_frames", split = "test")[:1500]
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


def save_dfs(val_dataset):
    name_list = ["white_male_conservative", "white_male_liberal", "white_female_liberal", "black_female_moderate_liberal", "white_female_conservative", "all"]
    url = "distilbert/distilroberta-base"
    for name in name_list:
        tokenizer = AutoTokenizer.from_pretrained(url)
        model = get_model(name, base_dir=Path("/Users/samuelwu/Desktop/Senior/Spring/DS682/Final Project/responsible-ai-audit/responsible_ai_audit/evaluate/models"))
        df = get_df_w_predictions(val_dataset,model,tokenizer)
        df.to_csv("data/"+name+"_data.csv")
    print("done")

def load_dfs_w_labels():
    name_list = ["white_male_conservative", "white_male_liberal", "white_female_liberal", "black_female_moderate_liberal", "white_female_conservative", "all"]
    dfs_w_labels = []
    for name in name_list:
        df = pd.read_csv("data/"+name+"_data.csv")
        dfs_w_labels.append((df, name))
    return dfs_w_labels
    
#bar plot
def bar_plots(dfs_with_label):
    for df_with_label in dfs_with_label:
        df = df_with_label[0]
        df = df['Prediction'].replace({0:'Offensive',1:'Neutral', 2:'Not Offensive'})
        label = df_with_label[1]
        fig = df.value_counts().plot(kind='bar')
        plt.title("Prediction Distribution: "+label)
        plt.savefig("assets/"+label+"_barplot.svg", dpi = 300)
        plt.show()

#plot TP/TN/FP/FN 

def FPR_bar_plot(dfs_with_labels):
    label = []
    value = []
    model = []
    for df,model_name in dfs_with_labels:
        model = model+ [model_name, model_name, model_name, model_name]
        label = label + ["TP", "TN", "FP", "FN"]
        value = value+[sum(offensiveYN == 1.0 and prediction == 0 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == 0.0 and prediction == 2 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == 0.0 and prediction == 0 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"])),
                       sum(offensiveYN == 1.0 and prediction == 2 for offensiveYN,prediction in zip(df["offensiveYN"], df["Prediction"]))]
    FPR_df = pd.DataFrame({"Label":label, "Value":value, "Model":model})
    plt.title("TP/TN/FP/FN Distribution")
    plt.xlabel("Model")
    plt.ylabel("# of Predictions")
    # Display the plot
    sns.barplot(x = 'Label',y = 'Value',hue = 'Model',data = FPR_df)
    plt.savefig("assets/FPR_barplot.svg", dpi = 300)
    plt.show()

def ROC_plots(dfs_with_label):
    for df_with_label in dfs_with_label:
        df = df_with_label[0]
        label = df_with_label[1]
        offYN = [1 if elem == 1.0 else 0 for elem in df["offensiveYN"]]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(offYN, df["OffensiveY Score"])
        roc_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
        plt.title("ROC Curve: "+label)
        plt.xlabel("FPR")
        plt.ylabel("TPR")

        # Display the plot
        sns.lineplot(data = roc_df, x = "fpr", y = "tpr", linewidth = 6)
        plt.savefig("assets/"+label+"_ROC_plot.svg", dpi = 300)
        plt.show()

def val_race_plot():
    df = pd.DataFrame(get_val_dataset())
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]
    #create pie chart
    plt.pie(df["annotatorRace"].value_counts(), colors = colors, autopct=lambda p: '{:.01f}%'.format(round(p)) if p > 1.0 else '', pctdistance=1.3, labels = None)
    plt.legend(df["annotatorRace"].unique(), bbox_to_anchor=(.7, -.05))
    plt.title("Validation Data Race %")
    plt.savefig("assets/val_race.svg",bbox_inches='tight', dpi = 300)
    plt.show()

def val_gender_plot():
    df = pd.DataFrame(get_val_dataset())
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]
    #create pie chart
    plt.pie(df["annotatorGender"].value_counts(), colors = colors, autopct=lambda p: '{:.01f}%'.format(round(p)) if p > 1.0 else '', pctdistance=.8, labels = None)
    plt.legend(df["annotatorGender"].unique(), bbox_to_anchor=(.7, -.05))
    plt.title("Validation Data Gender %")
    plt.savefig("assets/val_gender.svg",bbox_inches='tight', dpi = 300)
    plt.show()

def val_politics_plot():
    df = pd.DataFrame(get_val_dataset())
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]
    #create pie chart
    plt.pie(df["annotatorPolitics"].value_counts(), colors = colors, autopct=lambda p: '{:.01f}%'.format(round(p)) if p > 1.0 else '', pctdistance=.8, labels = None)
    plt.legend(df["annotatorPolitics"].unique(), bbox_to_anchor=(.7, -.05))
    plt.title("Validation Data Politics %")
    plt.savefig("assets/val_politics.svg",bbox_inches='tight', dpi = 300)
    plt.show()

#method to calculate the datasets and load them
#save_dfs(get_val_dataset())

#load and save figures
dfs = load_dfs_w_labels()
FPR_bar_plot(dfs)
ROC_plots(dfs)
bar_plots(dfs)
val_race_plot()
val_gender_plot()
val_politics_plot()


