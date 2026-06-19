import re
import os

IMAGE_ROOT = os.getenv("IMAGE_ROOT_DIR", "YOUR_IMAGE_ROOT_DIR")

# Dataset registry. Override DATA_ROOT and IMAGE_ROOT_DIR in your environment.
HALLU_ESTI_TRAINING = {
    "annotation_path": "datasets/VFM_train.json",
    "data_path": IMAGE_ROOT,
}

HALLU_ESTI_TRAINING_WRONGLESION = {
    "annotation_path": "datasets/VFM_train_wronglesion.json",
    "data_path": IMAGE_ROOT,
}

HALLU_ESTI_TRAINING_UNVISIBLE = {
    "annotation_path": "datasets/VFM_train_unvisible.json",
    "data_path": IMAGE_ROOT,
}

HALLU_ESTI_EVAL = {
    "annotation_path": "datasets/VFM_eval.json",
    "data_path": IMAGE_ROOT,
}

HALLU_ESTI_TEST = {
    "annotation_path": "datasets/HalluEsti_test.json",
    "data_path": IMAGE_ROOT,
}


data_dict = {
    'hallu_esti_training': HALLU_ESTI_TRAINING,
    'hallu_esti_wrong_lesion': HALLU_ESTI_TRAINING_WRONGLESION,
    'hallu_esti_training_unvisible': HALLU_ESTI_TRAINING_UNVISIBLE,
    'hallu_esti_eval': HALLU_ESTI_EVAL,
    'hallu_esti_test': HALLU_ESTI_TEST,
}


def parse_sampling_rate(dataset_name):  # 解析采样率，例如 "dataset%50" 表示 50% 的采样率
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):  # 根据数据集名称列表生成配置列表
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
