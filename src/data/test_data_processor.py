import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from ..utils.api_config import *
from ..utils.util import *
from ..cf_generate.gen_query import generate_supporting_and_counter_phrase
from ..eval.entity_extract import extract_entity_from_report

from .sys_message import SYS_MESSAGE

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


@dataclass
class TestDataBatch:
    sup_messages: List[Any]
    con_messages: List[Any]
    sup_entities: List[Any]
    con_entities: List[Any]
    is_hallu: List[Any]
    class_list: List[Any]
    gt_sup: List[Any]
    gt_con: List[Any]
    ori_bbox: List[Any]
    norm_bbox: List[Any]
    model_list: List[Any]
    image_list: List[Any]
    dataset_name_list: List[Any]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(
        f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(
        f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(
            f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(
        f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(
        f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(
            f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(
                f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(
            f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


# implemented
def _build_messages(item: Dict[str, Any], base_path: Path, mode: str, sys_enable: bool = False) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    # Add system message at the beginning of the conversation
    if sys_enable:
        messages.extend(SYS_MESSAGE)
    # build messages based on different mode
    for turn in item[mode]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":  # test dataset only has user messages
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    mode,
    sys_enable,
) -> Dict:  # preprocess a single data sample for Qwen-VL
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    assert mode in ["supporting_conversations",
                    "contradictory_conversations",
                    "extracted_entity_sup_conversations",
                    "extracted_entity_con_conversations"], f"mode {mode} not supported, must in ['supporting_conversations', 'contradictory_conversations']."

    source = sources[0]
    base_path = Path(source.get("data_path", ""))  # image root dir
    messages = _build_messages(source, base_path, mode, sys_enable)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", padding=True
    )  # process image and text and tokenize to input_ids

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    # no need for labels in test mode

    full_result["input_ids"] = input_ids
    return full_result


def _get_dataset(data_args):
    dataset = data_args.eval_dataset_use.split(",")  # split multiple datasets
    dataset_list = data_list(dataset)
    rank0_print(f"Loading datasets: {dataset_list}")

    list_data_dict = []

    for data in dataset_list:
        file_format = data['annotation_path'].split(".")[-1]

        # load json or jsonl annotation file
        if file_format == "jsonl":
            annotations = read_jsonl(data["annotation_path"])
        else:
            annotations = json.load(open(data["annotation_path"], "r"))

        # random sampling if sampling_rate < 1.0
        sampling_rate = data.get("sampling_rate", 1.0)
        if sampling_rate < 1.0:
            annotations = random.sample(
                annotations, int(len(annotations) * sampling_rate)
            )
            rank0_print(
                f"sampling {len(annotations)} examples from dataset {data}")
        else:
            rank0_print(f"dataset name: {data}")

        for ann in annotations['data']:
            if isinstance(ann, list):
                for sub_ann in ann:
                    sub_ann["data_path"] = data["data_path"]
            else:
                ann["data_path"] = data["data_path"]

        list_data_dict += annotations['data']

        rank0_print(
            f"Total test samples after loading dataset {data}: {len(list_data_dict)}")

    return list_data_dict


class LazySupervisedDataset_fortest(Dataset):
    """Dataset for supervised fine-tuning."""

    # input: image processor and data args
    def __init__(self, processor, list_data_dict, data_args):
        super(LazySupervisedDataset_fortest, self).__init__()

        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(
                f"model_type: {data_args.model_type} not supported")

        rank0_print("Formatting inputs...Skip in lazy mode")
        # update image processor args from data args
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split())
                    for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or (
                    "video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"]
                           for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(
                    f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:  # implemented
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            mode="extracted_entity_sup_conversations" if self.data_args.do_entity_extract else "supporting_conversations",
            sys_enable=self.data_args.sys_enable,
        )

        data_dict_cont = preprocess_qwen_visual(
            sources,
            self.processor,
            mode="contradictory_conversations",
            sys_enable=self.data_args.sys_enable,
        )

        # for supporting conversations
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        data_dict["label"] = sources[0].get("label", "")
        data_dict["cont_label"] = sources[0].get("cont_label", "")
        data_dict["class"] = sources[0].get("class", "")
        data_dict["image_list"] = sources[0].get("image", [])
        data_dict["video_list"] = sources[0].get("video", [])
        data_dict["bbox"] = sources[0].get("bbox", [])
        data_dict["norm_bbox"] = sources[0].get("norm_bbox", [])

        if "report" in sources[0]:
            data_dict["report"] = sources[0]["report"]
            # 1: support (default), 0: contradict
            data_dict["is_correct"] = sources[0].get("is_correct", 1)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            sources = [sources]
            assert len(
                sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                    # FIXME
                ), f"Don't know why it is wrapped to a list.\n {source}"
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"]
                                     for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }
            if all("labels" in d for d in data_list):
                new_data_dict["labels"] = torch.cat(
                    [d["labels"] for d in data_list], dim=1)

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # for supporting conversations
        input_ids, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        batch["label"] = [instance["label"] for instance in instances]
        batch["cont_label"] = [instance["cont_label"]
                               for instance in instances]
        batch["class"] = [instance["class"] for instance in instances]
        batch["image_list"] = [instance["image_list"]
                               for instance in instances]
        batch["video_list"] = [instance["video_list"]
                               for instance in instances]
        batch["bbox"] = [instance["bbox"] for instance in instances]
        batch["norm_bbox"] = [instance["norm_bbox"] for instance in instances]

        if "report" in instances[0]:
            reports = [instance["report"] for instance in instances]
            is_corrects = torch.tensor(
                [instance["is_correct"] for instance in instances],
                dtype=torch.long,
            )
            batch["report"] = reports
            batch["is_correct"] = is_corrects
        return batch


def make_supervised_data_module(processor, list_data_dict, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert data_args.eval_dataset_use != "", "eval_dataset_use must be specified."
    test_dataset = LazySupervisedDataset_fortest(
        processor, list_data_dict=list_data_dict, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        test_dataset=test_dataset, data_collator=data_collator
    )


def _do_entity_extraction(data_args, test_args):
    api_config = get_model_config(test_args.entity_extract_model)
    client = create_client(api_config.base_url, api_config.api_key)

    list_data_dict = _get_dataset(data_args)

    candidate_dict = load_json(test_args.entity_candidate_file)

    for item in tqdm(list_data_dict):
        report = item.get("report", "")
        report = report.lower()
        report = report.split("**findings:**")[-1].strip()

        entity = extract_entity_from_report(
            client, report, candidate_dict, model=test_args.entity_extract_model, api_config=api_config
        )

        item['extracted_entity'] = entity['entity']
        item['extracted_entity_type'] = entity['entity_type']

    return list_data_dict


def _build_conversations(entity, prompt, norm_bbox):
    conversation = [
        {
            "from": "human",
            "value": "<image>\n"+prompt,
        },
        {
            "from": "gpt",
            "value": '```json\n[\n\t{\"bbox_2d\": '+str(norm_bbox)+', \"label\": \"'+str(entity)+'\"}\n]\n```'
        }
    ]
    return conversation


def _create_counterfactuals(list_data_dict, do_entity_extraction, test_args):
    api_config = get_model_config(test_args.cf_model)
    client = create_client(api_config.base_url, api_config.api_key)

    for item in tqdm(list_data_dict):
        if do_entity_extraction:
            entity = item.get('extracted_entity', None)
            entity_type = item.get('extracted_entity_type', None)
            if entity is None:
                continue  # Skip if no entity extracted\
        else:
            if item.get('is_hallu', False):
                entity = item.get('con_phrase', None)
            else:
                entity = item.get('sup_phrase', None)
            entity_type = ''

            if entity is None:
                continue  # Skip if no entity available

        result = generate_supporting_and_counter_phrase(
            client, entity, entity_type, test_args.entity_candidate_file, test_args.cf_model)

        item['do_entity_extraction'] = do_entity_extraction
        item['extracted_entity_sup'] = result['supporting_phrase']
        item['extracted_entity_con'] = result['contradictory_phrase']
        item['extracted_entity_con_conversations'] = _build_conversations(
            result['contradictory_phrase'], result['contradictory_prompt'], item.get('norm_bbox', []))
        item['extracted_entity_sup_conversations'] = _build_conversations(
            result['supporting_phrase'], result['supporting_prompt'], item.get('norm_bbox', []))

    return list_data_dict


if __name__ == "__main__":
    pass
