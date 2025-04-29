# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os, glob
import random

def get_wav_filelist(data_root, data_type, subsample=1):
    wav_list = sorted(
        [
            path.replace(data_root, "")[1:]
            for path in glob.glob(os.path.join(data_root, data_type, "**/**/*.wav"), recursive=True)
        ]
    )
    wav_list = wav_list[::subsample]
    wav_list = [path.replace(".wav", "") for path in wav_list]

    return wav_list


def write_filelist(output_path, wav_list):
    with open(output_path, "w") as f:
        for i in range(len(wav_list)):
            filename = wav_list[i] + "|" # retain "|" to be consistent with original code
            f.write(filename + "\n")


if __name__ == "__main__":
    random.seed(42)

    data_root = "/data/berendhs"

    data_type_list = ["wcm90_segments_nonspeech"]
    subsample = [1]
    wav_list_all = []
    for data_type, subsample in zip(data_type_list, subsample):
        print(f"processing {data_type}")
        data_path = os.path.join(data_root, data_type)
        assert os.path.exists(data_path), (
            f"path {data_path} not found."
        )
        wav_list = get_wav_filelist(data_root, data_type, subsample=subsample)
        write_filelist(os.path.join(data_root, data_type + ".txt"), wav_list)
        wav_list_all.extend(wav_list)

    val_split, train_split = 0.005, 0.995
    assert val_split + train_split == 1, "Val and train split don't add up to one"
    random.shuffle(wav_list_all)
    n_train_files = int(len(wav_list_all) * train_split)
    print(f"Total number of files found is {len(wav_list_all)}")
    
    wav_list_train, wav_list_val =  wav_list_all[:n_train_files], wav_list_all[n_train_files:]
    print(f"Number of training files is: {len(wav_list_train)}")
    print(f"Number of validation files is: {len(wav_list_val)}")
    write_filelist(os.path.join(data_root, "train-full.txt"), wav_list_train)
    write_filelist(os.path.join(data_root, "val-full.txt"), wav_list_val)

    print("done")
