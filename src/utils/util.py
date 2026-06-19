# This script is implemented from https://github.com/zaixizhang/CBD/blob/main/utils/util.py
import re
import torch
import os
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import json
from glob import glob
from enum import Enum
import logging
import deepspeed
import ast
from PIL import Image
from itertools import combinations
from deepspeed.utils import logging as ds_logging

IGNORE_INDEX = -100
LEFT_BBOX_ID = 508
RIGHT_BBOX_ID = 1125


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return


def find_jsons(root_dir):
    return glob(os.path.join(root_dir, '*.json'))


def find_jsonls(root_dir):
    return glob(os.path.join(root_dir, '*.jsonl'))


def parse_json(json_output):  # @title Parsing JSON output
    if "```json" not in json_output.lower():
        return None
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i+1:])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found
    return json_output


def shutdown_ds_logging():
    # 1️⃣ 关掉 Python logging 层面的 deepspeed
    logging.getLogger("deepspeed").setLevel(logging.CRITICAL)
    logging.getLogger("deepspeed.runtime").setLevel(logging.CRITICAL)
    logging.getLogger("deepspeed.runtime.engine").setLevel(logging.CRITICAL)
    logging.getLogger("deepspeed.runtime.checkpoint_engine").setLevel(
        logging.CRITICAL)

    # 2️⃣ 关掉 DeepSpeed 自己的 log_dist / print_rank_*（关键）
    ds_logging.logger.setLevel(logging.CRITICAL)
    ds_logging.logging.disable(logging.INFO)


def get_image_token_mask(input_ids, tokenizer):
    """
    Get the mask for image tokens in the input_ids.
    """
    img_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    img_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    image_mask = torch.zeros_like(input_ids).bool()

    for b in range(input_ids.size(0)):
        start = (input_ids[b] == img_start_id).nonzero()[0].item()
        end = (input_ids[b] == img_end_id).nonzero()[0].item()
        image_mask[b, start + 1: end] = True

    return image_mask


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
        elif (
            isinstance(input_dict[k], dict)
            and len(input_dict[k]) > 0
            and isinstance(list(input_dict[k].values())[0], torch.Tensor)
        ):
            input_dict[k] = dict_to_cuda(v)
    return input_dict


def xyxy2xywh(bbox):
    """Convert bbox format from xyxy to xywh.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format

    Returns:
        Bounding box in [x, y, w, h] format
    """
    if isinstance(bbox, np.ndarray):
        _bbox = bbox.tolist()
    else:
        _bbox = bbox
    """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h] format."""
    return [_bbox[0], _bbox[1], _bbox[2] - _bbox[0], _bbox[3] - _bbox[1]]


def norm2abs_bbox(norm_bbox, width, height):
    """Convert normalized bbox to absolute bbox.

    Args:
        norm_bbox: Normalized bbox in [x1, y1, x2, y2] format (values between 0 and 1)
        width: Image width
        height: Image height

    Returns:
        Absolute bbox in [x1, y1, x2, y2] format
    """
    abs_x1 = int(norm_bbox[0]/1000 * width)
    abs_y1 = int(norm_bbox[1]/1000 * height)
    abs_x2 = int(norm_bbox[2]/1000 * width)
    abs_y2 = int(norm_bbox[3]/1000 * height)
    return [abs_x1, abs_y1, abs_x2, abs_y2]


def area_bbox(bbox):
    """Return the area of a bounding box."""
    if bbox[2] <= 0 or bbox[3] <= 0:
        return 0.0
    return float(bbox[2]) * float(bbox[3])


def iou_bboxes(bbox1, bbox2, format="xyxy"):
    """
    Compute IoU between two bounding boxes.

    Args:
        bbox1: [x, y, w, h] or [x1, y1, x2, y2]
        bbox2: same format as bbox1
        format: "xywh" or "xyxy"

    Returns:
        IoU (float)
    """

    if format == "xywh":
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2

        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2

    elif format == "xyxy":
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
    else:
        raise ValueError("format must be 'xywh' or 'xyxy'")

    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    # Areas
    area1 = max(0.0, (x2_1 - x1_1)) * max(0.0, (y2_1 - y1_1))
    area2 = max(0.0, (x2_2 - x1_2)) * max(0.0, (y2_2 - y1_2))

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def extract_bbox_from_text(text):
    pred_bbox = parse_json(text)

    if pred_bbox is not None:
        try:
            pred_bbox = ast.literal_eval(pred_bbox)
        except Exception as e:
            try:
                end_idx = pred_bbox.rfind('"}') + len('"}')
                truncated_text = pred_bbox[:end_idx] + "]"
                pred_bbox = ast.literal_eval(truncated_text)
            except Exception:
                pred_bbox = None

        if not isinstance(pred_bbox, list):
            pred_bbox = [pred_bbox]

    return pred_bbox


def cal_logits_uncertainty(generated_ids_trimmed, logits):
    results = []

    for b in range(len(generated_ids_trimmed)):
        gen_tokens = generated_ids_trimmed[b]

        # -------------------------------------------------
        # Step 1: 根据文本解析出 bbox 坐标，支持 [x,x,x,x] 或 JSON 格式
        # -------------------------------------------------
        left_pos = (gen_tokens == LEFT_BBOX_ID).nonzero(as_tuple=True)[0]
        right_pos = (gen_tokens == RIGHT_BBOX_ID).nonzero(as_tuple=True)[0]
        if len(left_pos) == 0 or len(right_pos) == 0:
            # 没有生成 bbox，直接跳过
            results.append(None)
            continue
        l = left_pos[0].item()
        r = right_pos[0].item()

        if l >= r:
            print(f"Invalid bbox format in generated tokens: {gen_tokens}")
            continue

        # -------------------------------------------------
        # Step 2: 根据位置找出对应的logits，计算置信度
        # -------------------------------------------------
        bbox_logits = logits[b, l:r+1, :]
        bbox_token_ids = gen_tokens[l:r+1]

        # 计算每个 bbox token 的置信度（取对应位置的 logit 中生成的 token 的概率）
        token_confidences = []
        for i, token_id in enumerate(bbox_token_ids):
            token_logit = bbox_logits[i]
            token_prob = F.softmax(token_logit, dim=-1)[token_id].item()
            token_confidences.append(token_prob)

        if len(token_confidences) > 0:
            confidence = float(torch.tensor(token_confidences).mean())
        else:
            confidence = None

        results.append(confidence)

    return results


def cal_sample_uncertainty(model, processor, inputs, batch_images, image_root, num_samples=10, temperature=0.7, top_p=0.9):
    """
    Multi-sample stability uncertainty for Qwen-VL bbox grounding

    Returns:
        List[dict]
    """
    model.eval()

    all_sample_boxes = [[] for _ in range(inputs.input_ids.shape[0])]
    all_sample_texts = [[] for _ in range(inputs.input_ids.shape[0])]

    # multi sampling
    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=128,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        # trim
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(
                inputs.input_ids, outputs
            )
        ]

        decoded = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # get bbox
        for b, text in enumerate(decoded):
            image_path = os.path.join(image_root, batch_images[b])
            image = Image.open(image_path)
            width, height = image.size
            pred_bbox = extract_bbox_from_text(text)
            # if bbox is invalid, use [0,0,1,1] as default to avoid affecting IoU calculation
            pred_bbox_abs = norm2abs_bbox(
                pred_bbox[0]["bbox_2d"], width, height) if (pred_bbox is not None) and (pred_bbox[0] is not None) and (len(pred_bbox) != 0) else [0, 0, 1, 1]
            all_sample_boxes[b].append(pred_bbox_abs)
            all_sample_texts[b].append(text)

    # calculate uncertainty (IoU variance)
    results = []
    for b in range(inputs.input_ids.shape[0]):
        sample_bboxes = all_sample_boxes[b]

        pairs = list(combinations(sample_bboxes, 2))
        ious = [iou_bboxes(a, c) for a, c in pairs]
        stability = sum(ious)/len(ious) if len(ious) > 0 else 0.0

        results.append({
            "sample_bboxes": sample_bboxes,
            "stability_confidence": float(stability),
        })

    return results


def safe_mean(values):
    nums = [v for v in values if v is not None]
    return sum(nums)/len(nums) if len(nums) > 0 else None


def cal_uncertainty_score(sup_bbox, sup_sample_uncertainty, con_bbox, con_sample_uncertainty):
    """
    Calculate the final uncertainty score by combining supervised and counterfactual uncertainties.

    Returns:
        float: Final uncertainty score (between 0 and 1)
    """
    if sup_bbox is not None and con_bbox is not None and sup_sample_uncertainty is not None and con_sample_uncertainty is not None:
        # Both uncertainties are available, take the average
        iou = iou_bboxes(sup_bbox, con_bbox)
        # uncertainty = iou * \
        #     ((sup_sample_uncertainty + con_sample_uncertainty) / 2)
        uncertainty = sup_sample_uncertainty-iou*con_sample_uncertainty
        return uncertainty, iou
    elif sup_bbox is not None:
        # Only supporting uncertainty is available
        return 1.0, None
    elif con_bbox is not None:
        # Only counterfactual uncertainty is available
        return 0.0, None
    else:
        # No uncertainty available, return None or a default value
        return 0.0, None


def cal_uncertainty_score_no_iou(sup_bbox, sup_sample_uncertainty, con_bbox, con_sample_uncertainty):
    """
    Calculate the final uncertainty score by combining supervised and counterfactual uncertainties.

    Returns:
        float: Final uncertainty score (between 0 and 1)
    """
    if sup_bbox is not None and con_bbox is not None and sup_sample_uncertainty is not None and con_sample_uncertainty is not None:
        # Both uncertainties are available, take the average
        iou = iou_bboxes(sup_bbox, con_bbox)
        # uncertainty = iou * \
        #     ((sup_sample_uncertainty + con_sample_uncertainty) / 2)
        uncertainty = sup_sample_uncertainty-con_sample_uncertainty
        return uncertainty, iou
    elif sup_bbox is not None:
        # Only supporting uncertainty is available
        return 1.0, None
    elif con_bbox is not None:
        # Only counterfactual uncertainty is available
        return 0.0, None
    else:
        # No uncertainty available, return None or a default value
        return 0.0, None
