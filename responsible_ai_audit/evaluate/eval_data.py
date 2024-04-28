#loading the datasets from load_data.py
import load_data

#plot the annotatorGender dist
def genderDist():
    data = load_data.getRandDataset(10)
    print(data)

#plot annotatorRace dist
def raceDist():
    data = load_data.getRandDataset(10)

#plot annotatorPolitics dist
def poliDist():
    data = load_data.getRandDataset(10)
