from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

dataset = load_dataset("social_bias_frames")

def get_datasets(dataset):
    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]

    train.set_format("torch")
    validation.set_format("torch")
    test.set_format("torch")

    # train_loader = DataLoader(train, batch_size=32, shuffle=True)
    # validation_loader = DataLoader(validation, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test, batch_size=32, shuffle=True)

    return train, validation, test

# other populations to explore - hisp man: 3k - liberal: 900, cons-1.8k, hisp woman: 4k - mod-liberal: 1k, other: 2k

# white male conservative - train/val/test = 7215/1005/1085
def getWhiteMaleConsData():
    white_male_cons = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                        example['annotatorGender'] == 'man' and
                                                        example['annotatorPolitics'] == 'cons')
    return get_datasets(white_male_cons)

# white male liberal - train/val/test = 16,051/2557/2445    
def getWhiteMaleLibData():
    white_male_liberal = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                        example['annotatorGender'] == 'man' and
                                                        example['annotatorPolitics'] == 'liberal')
    return get_datasets(white_male_liberal)

# white female liberal - train/val/test = 23,525/3414/3568
def getWhiteFemaleLibData():
    white_female_liberal = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                      example['annotatorGender'] == 'woman' and
                                                      example['annotatorPolitics'] == 'liberal')
    return get_datasets(white_female_liberal)

# white female conservative - train/val/test = 997/186/183
def getWhiteFemaleConsData():
    white_female_cons = dataset.filter(lambda example: example['annotatorRace'] == 'white' and
                                                    example['annotatorGender'] == 'woman' and
                                                    example['annotatorPolitics'] == 'cons')
    return get_datasets(white_female_cons)

# black female mod-liberal - train/val/test = 4009/593/645
def getBlackFemaleModlibData():
    black_female_modliberal = dataset.filter(lambda example: example['annotatorRace'] == 'black' and
                                                            example['annotatorGender'] == 'woman' and
                                                            example['annotatorPolitics'] == 'mod-liberal')
    return get_datasets(black_female_modliberal)