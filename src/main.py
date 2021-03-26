from pathlib import Path
import librosa
import os
import os.path
import wave

"""
Utility
        returning a directory of dataset
Input
        @test : a flag
                if test == 0, 
                    get 'training' dataset
                else 
                    get 'testing' dataset
Output
        a directory,
        key     : integers, means labels
        value   : a list including many numpy.array elements, 
                    every element is a single corpus data.
"""
def prepareDataset(test = 0):
    dataset_path = Path("corpus/train")
    if test:
        dataset_path = Path("corpus/test")
    label_dirs = os.listdir(dataset_path)
    dataset = {}
    for label in label_dirs:
        dataset[int(label)] = []
        corpus = os.listdir(dataset_path / label)
        for single_corpus in corpus:
            audio_data, _ = librosa.load(dataset_path / label / single_corpus, sr=None)
            dataset[int(label)].append(audio_data)
    return dataset

"""
Utility
        converting a dataset to a 'mfcc' dataset
Input
        @dataset : a directory produced from prepareDataset()
Output
        a dataset with mfcc values

        changing the input 'in-place'.
"""
def convertDatasetToMFCC(dataset):
    pass


### main function ###
def main():
    train_dataset = prepareDataset()
    for i in train_dataset:
        print(train_dataset[i])

    """
    test_dataset = prepareDataset(1)
    for i in test_dataset:
        print(test_dataset[i])
    """

if __name__ == "__main__":
    main()