import os
from math import ceil

import librosa
import numpy as np
import pandas as pd

import feature_config as fc


def read_flusense_meta(meta_csv, exclude):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    labels = []
    intervals = []
    count = 0

    for row in df.iterrows():
        file_name = row[1]['filename']
        label = row[1]['label']
        interval = [row[1]['start'], row[1]['end']]

        # Skip the audio intervals with length smaller than 0.5s
        # Skip the audios with labels in the exclude (list)
        dura = interval[1] - interval[0]
        if dura < 0.5 or label in exclude:
            continue
        c1, c2 = divmod(dura, 1)
        count += c1
        if c2 >= 0.5:
            count += 1

        audio_names.append(file_name)
        labels.append(label)
        intervals.append(interval)

    return audio_names, labels, intervals, int(count)


def read_dicova_meta(meta_csv):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    covid_statuses = []
    genders = []

    for row in df.iterrows():
        file_name = row[1]['uuid']
        covid_status = row[1]['status']
        covid_status = 0 if covid_status == 'healthy' else 1
        gender = row[1]['gender']

        audio_names.append(file_name)
        covid_statuses.append(covid_status)
        genders.append(gender)

    return audio_names, covid_statuses, genders


def read_compare_metadata(meta_csv):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    labels = []

    for row in df.iterrows():
        file_name = row[1]['filename']
        label = row[1]['label']
        label = 0 if label == 'negative' else 1

        audio_names.append(file_name)
        labels.append(label)

        # if len(audio_names) > 8:
        #     return audio_names, labels

    return audio_names, labels


#######################################


def get_file_path(dataset_path, filename):
    if '.wav' in filename:
        return os.path.join(dataset_path, filename)
    return os.path.join(dataset_path, filename + '.wav')


def get_wav_duration(file_path, mode='length'):
    wav_info = librosa.load(file_path, sr=None)
    if 'length' in mode:
        return len(wav_info[0]), wav_info[1]
    return len(wav_info[0]) / wav_info[1], wav_info[1]


def get_segment_num(data_length, segment_duration, threshold, sample_rate):
    quotient, mod = divmod(data_length / sample_rate, segment_duration)
    if mod < threshold:
        return quotient
    return quotient + 1


def load_audio_data(dataset_path, filename):
    filepath = get_file_path(dataset_path, filename)
    wav_data, sr = librosa.load(filepath, sr=None)
    return wav_data, sr


def is_padding(audio_length, seg_num, seg_length):
    if audio_length < seg_num * seg_length:
        return True
    return False


def pad_data(audio_data, target_length, mode='cycle'):
    reps = ceil(target_length / len(audio_data))
    if 'cycle' in mode:
        audio_data = np.tile(audio_data, reps=reps)[:target_length]
    elif 'zero' in mode:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant', constant_values=(0, 0))
    else:
        raise Exception('Unexpect mode! Use \'cycle\' or \'zero\' in mode')

    return audio_data


def spilt_data(audio_data, seg_length, seg_num):
    seg_num = int(seg_num)
    for idx in range(seg_num):
        start = idx * seg_length
        yield audio_data[start: start + seg_length]


def get_melspec_width(audio_length, center=False):
    """
    :param audio_length: int
    :param center: Boolean, default is False
            - if True, the signal y is padded so that frame t is centered at y[t * hop_length].
            - If False, then frame t begins at y[t * hop_length]
    :return: the width of the melspectrogram

    refer: https://github.com/librosa/librosa/issues/530
    """
    if center:
        return audio_length // fc.hop_length + 1
    else:
        return (audio_length - fc.n_fft) // fc.hop_length + 1


def calculate_melspec_librosa(audio_data, sample_rate, mode='max', center=False):
    """
    :param mode: max & one, logmel = 10 * log10(S / ref)
            - if max, divide the maximum value, ref = np.max
            - if one, abort normalization, ref = 1
    :param center: check the get_melspec_width()
    :return: the result will be transposed to make the n_mels as the last dimension
    """
    if 'max' in mode:
        ref = np.max
    elif 'one' in mode:
        ref = 1
    else:
        raise Exception('Wrong Mode! Please use \'max\' or \'one\'')

    mel_spect = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=fc.n_fft,
                                               hop_length=fc.hop_length, n_mels=fc.n_mels, center=center)
    result = librosa.power_to_db(mel_spect, ref=ref)
    return result.T


def calculate_delta_librosa(data, order=1):
    result = librosa.feature.delta(data, order=order)
    return result


def print_progress(idx, len, base):
    if idx % base == base - 1:
        print('{:.2%}'.format(idx / len))
