from .cal_metric import xyxy2xywh, norm2abs_bbox, iou_bboxes, area_bbox, cal_metrics_miou
from transformers.trainer_utils import EvalPrediction
from typing import Dict, Optional
import numpy as np
import torch
import json
import ast
import os
from PIL import Image
import re


def parse_prediction_bbox(pred_text: str) -> Optional[list]:
    try:
        pred_bbox = json.loads(pred_text)
        if isinstance(pred_bbox, list) and len(pred_bbox) > 0:
            if isinstance(pred_bbox[0], dict) and "bbox_2d" in pred_bbox[0] and len(pred_bbox[0]["bbox_2d"]) == 4:
                return pred_bbox[0]["bbox_2d"]
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        try:
            end_idx = pred_text.rfind('"}') + len('"}')
            if end_idx > 0:
                truncated_text = pred_text[:end_idx] + "]"
                pred_bbox = json.loads(truncated_text)
                if isinstance(pred_bbox, list) and len(pred_bbox) > 0:
                    if isinstance(pred_bbox[0], dict) and "bbox_2d" in pred_bbox[0] and len(
                            pred_bbox[0]["bbox_2d"]) == 4:
                        return pred_bbox[0]["bbox_2d"]
        except (json.JSONDecodeError, TypeError, KeyError, IndexError, ValueError):
            pass
    return None


def compute_metrics(eval_prediction: EvalPrediction, compute_result: bool = True) -> Dict[str, float]:
    """
    严格适配 Trainer.evaluation_loop 调用方式
    签名必须匹配: (EvalPrediction, compute_result: bool = False) -> Dict[str, float]
    """
    print(EvalPrediction)
    global processor, image_root

    # # 1. 在batch模式下，如果不是最后一步，跳过完整计算
    # if not compute_result:
    #     return {"batch_processed": len(eval_prediction.label_ids) if eval_prediction.label_ids is not None else 0}

    # 2. 检查 processor 是否已设置
    if processor is None:
        print("⚠️ Processor not set in compute_metrics, returning dummy metrics")
        return {"miou": 0.0}

    processor = processor
    image_root = image_root

    # 3. 获取预测
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    # 4. 处理不同格式的预测
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # 取 logits 部分

    # 5. 转换为 numpy 数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 6. 从 logits 获取 token IDs
    if predictions.ndim == 3:  # [batch_size, seq_len, vocab_size]
        pred_token_ids = np.argmax(predictions, axis=-1)
    else:
        pred_token_ids = predictions

    # 7. 从 inputs 获取必要数据
    inputs = getattr(eval_prediction, "inputs", None)
    if inputs is None:
        inputs = getattr(eval_prediction, "input_ids", None)  # 兼容旧版本

    # 8. 获取图像路径和 bbox
    image_paths = []
    gt_bboxes = []

    if inputs is not None:
        if isinstance(inputs, dict):
            if "image" in inputs:
                image_paths = inputs["image"]
            if "bbox" in inputs:
                gt_bboxes = inputs["bbox"]
                if isinstance(gt_bboxes, torch.Tensor):
                    gt_bboxes = gt_bboxes.cpu().numpy()
        elif hasattr(inputs, "image"):
            image_paths = inputs.image
        elif hasattr(inputs, "bbox"):
            gt_bboxes = inputs.bbox.cpu().numpy() if torch.is_tensor(
                inputs.bbox) else inputs.bbox

    # 9. 处理样本 (限制数量以提高速度)
    max_samples = min(10, len(pred_token_ids))
    pred_bboxes = []
    gt_bbox_list = []

    for i in range(max_samples):
        if i >= len(image_paths) or i >= len(gt_bboxes):
            continue

        try:
            # 解码预测文本
            text = processor.tokenizer.decode(
                pred_token_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            # 解析预测 bbox
            pred_bbox = parse_prediction_bbox(text)
            print("pred_bbox:"+str(pred_bbox))
            if pred_bbox is None:
                continue

            # 获取图像尺寸
            img_path = os.path.join(image_root, image_paths[i])
            with Image.open(img_path) as img:
                width, height = img.size

            # 转换为绝对坐标
            pred_bbox_abs = norm2abs_bbox(pred_bbox, width, height)

            # 获取 ground truth bbox
            gt_bbox = gt_bboxes[i]
            if isinstance(gt_bbox, torch.Tensor):
                gt_bbox = gt_bbox.cpu().numpy()
            if not isinstance(gt_bbox, (list, tuple, np.ndarray)):
                continue

            gt_bbox = np.array(gt_bbox).flatten()[:4]  # 确保是4个值

            pred_bboxes.append(pred_bbox_abs)
            gt_bbox_list.append(gt_bbox.tolist() if isinstance(
                gt_bbox, np.ndarray) else gt_bbox)
        except Exception as e:
            continue

    # 10. 计算 mIoU
    if pred_bboxes and gt_bbox_list:
        try:
            miou, _ = cal_metrics_miou(pred_bboxes, gt_bbox_list)
        except Exception as e:
            miou = 0.0
    else:
        miou = 0.0
    print("eval_miou："+str(miou))
    return {"eval_miou": float(miou)}

# 全局变量设置函数（在训练开始前调用）


def set_global_vars(global_processor, global_image_root=None):
    """设置 compute_metrics 需要的全局变量"""
    global processor, image_root
    processor = global_processor
    image_root = global_image_root
    print("✅ Global variables set for compute_metrics")
