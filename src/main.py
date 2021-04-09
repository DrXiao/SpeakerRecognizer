from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from keras.preprocessing import sequence
from hmmlearn import hmm
import librosa
from scipy.fftpack import dct
import os
import os.path
import wave
import math


def extract_mfcc(path):
    # File assumed to be in the same directory
    signal, sample_rate = librosa.load(path, mono=True)
    signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    pre_emphasis = 0.95
    emphasized_signal = np.append(
        signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * \
        sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(
        float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[
        :, 1: (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    cep_lifter = 22
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  # *

    return mfcc


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


def prepareDataset(dataset_path):
    label_dirs = os.listdir(dataset_path)
    dataset = {}
    for label in label_dirs:
        dataset[int(label)] = []
        corpus = os.listdir(dataset_path / label)
        for single_corpus in corpus:
            audio_data = extract_mfcc(dataset_path / label / single_corpus)
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


def convertDatasetToMFCC(dataset, sampling_rate):
    for label in dataset:
        for corpus_idx in range(len(dataset[label])):
            dataset[label][corpus_idx] = librosa.feature.mfcc(
                y=dataset[label][corpus_idx], sr=sampling_rate, n_mfcc=40)
            # dataset[label][corpus_idx] = np.reshape(dataset[label][corpus_idx], (370, 14))


"""
Utility
        get kmeans parameter
Input
        @train_dataset : a directory with audio data converted to mfcc sequence
Output
        a sequence of kmeans points.
"""


def kmeansCenter(train_dataset):
    train_data = []
    for label in train_dataset:
        train_data += train_dataset[label]
    train_data = np.vstack(train_data)
    clusters = len(train_dataset.keys()) + 1
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(train_data)
    centers = kmeans.cluster_centers_
    return centers


"""
Utility
        Creating HMM models
Input
        @train_dataset : a directory with mfcc sequences
        @kmeans_centers : centers of kmeans
Output
        a directory
        key     : label
        value   : HMM model
"""


def createHMM_Model(train_dataset, kmeans_centers):
    hmm_model = {}
    states_num = 6

    for train_label in train_dataset:
        model = hmm.MultinomialHMM(
            n_components=states_num, n_iter=20, algorithm='viterbi', tol=0.01)
        train_data = train_dataset[train_label]
        train_data = np.vstack(train_data)
        train_data_label = []
        for corpus_idx in range(len(train_data)):
            dic_min = np.linalg.norm(
                train_data[corpus_idx] - kmeans_centers[0])
            label = 0
            for centers_idx in range(len(kmeans_centers)):
                if np.linalg.norm(train_data[corpus_idx] - kmeans_centers[centers_idx]) < dic_min:
                    dic_min = np.linalg.norm(
                        train_data[corpus_idx] - kmeans_centers[centers_idx]) < dic_min
                    label = centers_idx
            train_data_label.append(label)

        train_data_label.append(len(kmeans_centers) - 1)
        train_data_label = np.array([train_data_label])
        # print(train_data_label)
        model.fit(train_data_label)
        hmm_model[train_label] = model

    return hmm_model


### main function ###
def main():
    train_dataset = prepareDataset(Path("corpus/train"))
    #convertDatasetToMFCC(train_dataset, train_sampling_rate)

    kmeans_centers = kmeansCenter(train_dataset)

    hmm_models = createHMM_Model(train_dataset, kmeans_centers)
    print("Finish training of the HMM models")
    test_dataset = prepareDataset(Path("corpus/test"))
    #convertDatasetToMFCC(test_dataset, test_sampling_rate)
    true = []
    pred = []
    score_cnt = 0
    corpus_num = 0
    for test_label in test_dataset:
        feature = test_dataset[test_label]
        corpus_num += len(feature)
        for corpus_idx in range(len(feature)):
            test_data_label = []
            for j in range(len(feature[corpus_idx])):
                dic_min = np.linalg.norm(
                    feature[corpus_idx][j] - kmeans_centers[0])
                predict_label = 0
                for centers_idx in range(len(kmeans_centers)):
                    if np.linalg.norm(feature[corpus_idx][j] - kmeans_centers[centers_idx]) < dic_min:
                        dic_min = np.linalg.norm(
                            feature[corpus_idx][j] - kmeans_centers[centers_idx])
                        predict_label = centers_idx
                test_data_label.append(predict_label)
            test_data_label = np.array([test_data_label])
            # print(test_data_label)
            score_list = {}
            for model_label in hmm_models:
                model = hmm_models[model_label]

                score = model.score(test_data_label)
                score_list[model_label] = math.exp(score)
            predict_label = max(score_list, key=score_list.get)
            # print(score_list)
            # print("Test on true label ", test_label, ": predict result label is ", predict_label)
            if test_label == predict_label:
                score_cnt += 1
            true.append(test_label)
            pred.append(predict_label)
    print("true:", true, "pred:", pred, sep='\n')
    print("Final recognition rate is %.2f%%" %
          (100.0 * score_cnt/corpus_num))
    #with open("prob.txt", "a") as file:
        #file.writelines("Final recognition rate is %.2f%%\n" %
        #                (100.0 * score_cnt/corpus_num))


if __name__ == "__main__":
    for i in range(10):
        main()
