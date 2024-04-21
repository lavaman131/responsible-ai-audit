#this is a dummy program that loads the hate_speech_dataset and returns a random subset as a pandas dataset
from datasets import load_dataset
import pandas as pd

def getRandDataset(n):
    dataset = load_dataset("social_bias_frames", split="train")
    shuffled_dataset = dataset.shuffle(seed=42)
    subset = shuffled_dataset[0:n]
    return subset


