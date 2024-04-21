#loading the datasets
from datasets import load_dataset
dataset = load_dataset("social_bias_frames", split="train")

print(dataset[0])