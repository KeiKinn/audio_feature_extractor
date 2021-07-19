import argparse
import os
import time

import h5py
import numpy as np

import feature_config as fc
import feature_utils as fu


def feature_extractor(args):
    workspace = args.workspace
    dataset_dir = args.dataset_dir

    seg_duration = fc.seg_duration
    threshold = fc.threshold

    dataset_path = os.path.join(dataset_dir, 'audio')
    meta_path = os.path.join(dataset_dir, 'meta.csv')
    hdf5_path = os.path.join(workspace, 'testing_features.hdf5')
    wavs_length = []
    seg_num = []

    time_begin = time.time()

    filenames, labels, _ = fu.read_meta(meta_path)

    for filename in filenames:
        filepath = fu.get_file_path(dataset_path, filename)
        wav_length, sr = fu.get_wav_duration(filepath, mode='length')
        wavs_length.append(wav_length)
        seg_num.append(int(fu.get_segment_num(wav_length, seg_duration, threshold, sr)))

    seg_length = sr * seg_duration
    audios_sum = len(filenames)
    seg_sum = int(sum(seg_num))
    print('#audios in meta csv: {}, #segments is: {}'.format(audios_sum, seg_sum))

    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset(name='audio_name', shape=(seg_sum,), dtype='S80')
        hf.create_dataset(name='label', shape=(seg_sum,), dtype='S80')
        hf.create_dataset(name='logmel', shape=(seg_sum, seg_length / fc.hop_length - 1, fc.n_mels),
                          dtype=np.float32)

        idy = 0
        for idx in range(audios_sum):
            fu.print_progress(idx, audios_sum, fc.base_number)
            wav_data, sr = fu.load_audio_data(dataset_path, filenames[idx])
            ispadding = fu.is_padding(wavs_length[idx], seg_num[idx], seg_length)

            if ispadding:
                target_length = seg_num[idx] * seg_length
                wav_data = fu.pad_data(wav_data, target_length)

            seg_data = fu.spilt_data(wav_data, seg_length, seg_num[idx])
            for i, seg in enumerate(seg_data):
                hf['audio_name'][idy] = filenames[idx].encode()
                hf['label'][idy] = labels[idx]
                hf['mel_spec'][idy] = fu.calculate_melspec_librosa(seg, sr)
                idy += 1
    time_elapsed = time.time() - time_begin
    print('Done! Features extraction completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('And Features have been saved at {}'.format(hdf5_path))


if __name__ == '__main__':
    # this part is for debugging
    DATASET_DIR = '../../workspace'
    WORKSPACE = '../../workspace'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='logmel')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--workspace', type=str, default=WORKSPACE)
    args = parser.parse_args()
    if args.mode == 'logmel':
        feature_extractor(args)
    else:
        raise Exception('Incorrect arguments!')
