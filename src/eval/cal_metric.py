import os
import json
import math
from tqdm import tqdm
from glob import glob
from src.utils.util import *
import numpy as np
import ast
from PIL import ImageColor
from PIL import Image, ImageDraw, ImageFont
import argparse
from src.utils.util import *


def cal_metrics_miou(pred_bboxes, gt_bboxes, valid_mask):
    """Calculate mIoU between pred bboxes and gt bboxes.

    Args:
        pred_bboxes (list/np.ndarray)
        gt_bboxes (list/np.ndarray)
        valid_mask (list of bool): Mask indicating valid bbox pairs

    Returns:
        miou (float)
        iou_list (list of float): IoU for each bbox pair
    """
    if isinstance(pred_bboxes, np.ndarray):
        pred_bboxes = pred_bboxes.tolist()
    if isinstance(gt_bboxes, np.ndarray):
        gt_bboxes = gt_bboxes.tolist()

    assert len(pred_bboxes) == len(
        gt_bboxes), "The number of predicted bboxes and gt bboxes must be the same."

    total_num = len(valid_mask)

    iou_sum = 0.0
    iou_list = []
    for i in range(len(pred_bboxes)):
        pred_bboxes_xywh = xyxy2xywh(pred_bboxes[i])
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes[i])
        iou = iou_bboxes(pred_bboxes_xywh, gt_bboxes_xywh)
        iou_list.append(iou)
        iou_sum += iou

    miou = iou_sum / total_num if total_num > 0 else 0.0
    miou_only_valid = iou_sum / \
        len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    return miou, miou_only_valid, iou_list, total_num


def delete_classname_with_and(result_root):
    """Delete 'class' field in result json files if it contains 'and'.

    Args:
        result_root (str): Root directory of result JSON files.
    """
    result_files = glob(os.path.join(result_root, "*.json"))

    for result_file in tqdm(result_files, desc="Deleting 'and' in class names"):
        results = load_json(result_file)

        filtered_results = []
        for res in results:
            class_name = res.get("class", "")
            if "_and_" not in class_name.lower():
                filtered_results.append(res)

        save_json(filtered_results, result_file)


def cal_metrics_imis_full(result_root, save_file, image_root, uncertainty_type='none', model_name="Qwen3-VL-4B-Instruct", save_iou=True):
    """Calculate mIoU for IMIS dataset full set.

    Args:
        result_root (str): Root directory of result JSON files.
        save_file (str): Path to save the mIoU results.
        save_iou (bool): Whether to save individual IoU scores.
    """
    result_files = glob(os.path.join(result_root, "*.json"))
    all_mious = []
    all_mious_only_valid = []
    all_iou_lists = []
    all_logits_uncertainties = []
    all_sample_uncertainties = []
    total_nums = []
    per_dataset_mious = {}

    for result_file in tqdm(result_files, desc="Calculating mIoU for IMIS full set"):
        results = load_json(result_file)
        dataset_name = result_file.split(
            "/")[-1].split(f"_{model_name}_results.json")[0]
        if "finding-lungs-in-ct-data_3d" in dataset_name:
            continue
        pred_bboxes = []
        gt_bboxes = []
        valid_mask = []
        logits_uncertainties = []
        sample_uncertainties = []
        for res in results:
            image_path = os.path.join(image_root, res["image"])
            image = Image.open(image_path)
            width, height = image.size
            pred_str = res["prediction"]
            pred_bbox = parse_json(pred_str)
            gt_bbox = res["bbox"]
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

                if pred_bbox is None:
                    valid_mask.append(False)
                    continue

                if not isinstance(pred_bbox, list):
                    pred_bbox = [pred_bbox]

                if len(pred_bbox) == 0:
                    valid_mask.append(False)
                    continue

                try:
                    if len(pred_bbox[0].get("bbox_2d", [])) != 4:
                        valid_mask.append(False)
                        continue
                except Exception as e:
                    print(result_file)
                    valid_mask.append(False)
                    continue
                try:
                    pred_bbox_abs = norm2abs_bbox(
                        pred_bbox[0]["bbox_2d"], width, height)
                except Exception as e:
                    valid_mask.append(False)
                    continue
                pred_bboxes.append(pred_bbox_abs)
                gt_bboxes.append(gt_bbox)
                valid_mask.append(True)
                res["pred_bbox_abs"] = pred_bbox_abs
                res["pred_bbox_norm"] = pred_bbox[0]["bbox_2d"]
            else:
                valid_mask.append(False)

            if uncertainty_type == 'logits':
                logits_uncertainties.append(
                    res.get("logits_uncertainty", None))
            elif uncertainty_type == 'sample':
                sample_un = res.get("sample_uncertainty", None)
                if sample_un == 1.0:
                    sample_un = None
                sample_uncertainties.append(sample_un)
            elif uncertainty_type == 'all':
                logits_uncertainties.append(
                    res.get("logits_uncertainty", None))
                sample_un = res.get("sample_uncertainty", None)
                if sample_un == 1.0:
                    sample_un = None
                sample_uncertainties.append(sample_un)

        miou, miou_only_valid, iou_list, total_num = cal_metrics_miou(
            pred_bboxes, gt_bboxes, valid_mask)
        all_mious.append(miou)
        all_mious_only_valid.append(miou_only_valid)
        all_iou_lists.extend(iou_list)
        total_nums.append(total_num)
        mean_logits_uncertainty = safe_mean(
            logits_uncertainties)
        mean_sample_uncertainty = safe_mean(
            sample_uncertainties)
        all_logits_uncertainties.extend(logits_uncertainties)
        all_sample_uncertainties.extend(sample_uncertainties)
        per_dataset_mious[dataset_name] = {
            "miou": miou,
            "miou_only_valid": miou_only_valid,
            "valid_num": len(pred_bboxes),
            "total_num": total_num,
            "mean_logits_uncertainty": mean_logits_uncertainty,
            "mean_sample_uncertainty": mean_sample_uncertainty,
            "valid_ratio": len(pred_bboxes)/total_num if total_num > 0 else 0.0
        }

        if save_iou:
            iou_idx = 0

            for i, res in enumerate(results):
                if valid_mask[i]:
                    res["iou"] = iou_list[iou_idx]
                    iou_idx += 1
                else:
                    res["iou"] = None

            save_json(results, result_file)

    mean_iou = sum(all_mious) / \
        len(all_mious) if len(all_mious) > 0 else 0.0
    mean_iou_only_valid = sum(all_mious_only_valid) / \
        len(all_mious_only_valid) if len(all_mious_only_valid) > 0 else 0.0
    micro_iou = sum(all_iou_lists) / \
        sum(total_nums) if sum(total_nums) > 0 else 0.0
    micro_iou_only_valid = sum(
        all_iou_lists)/len(all_iou_lists) if len(all_iou_lists) > 0 else 0.0

    mean_logits_uncertainty = safe_mean(
        all_logits_uncertainties)
    mean_sample_uncertainty = safe_mean(
        all_sample_uncertainties)

    print(f"Overall mIoU (macro): {mean_iou:.4f}")
    print(f"Overall mIoU Only Valid (macro): {mean_iou_only_valid:.4f}")
    print(f"Overall mIoU (micro): {micro_iou:.4f}")
    print(f"Overall mIoU Only Valid (micro): {micro_iou_only_valid:.4f}")

    save_data = {
        "overall_miou_macro": mean_iou,
        "overall_miou_only_valid_macro": mean_iou_only_valid,
        "overall_miou_micro": micro_iou,
        "overall_miou_only_valid_micro": micro_iou_only_valid,
        "mean_logits_uncertainty": mean_logits_uncertainty,
        "mean_sample_uncertainty": mean_sample_uncertainty,
        "per_dataset_mious": per_dataset_mious
    }

    save_path = os.path.join(result_root, 'metrics', save_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"Saved mIoU results to {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str,
                        required=True)
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument("--uncertainty_type", type=str,
                        default="none", choices=["logits", "sample", "all", "none"])
    parser.add_argument("--image_root", type=str,
                        default=os.getenv("IMAGE_ROOT_DIR", "YOUR_IMAGE_ROOT_DIR"))
    args = parser.parse_args()

    save_file = "imis_full_miou_results_partial.json"
    delete_classname_with_and(args.result_root)
    cal_metrics_imis_full(args.result_root, save_file, args.image_root, uncertainty_type=args.uncertainty_type,
                          model_name=args.version, save_iou=True)
