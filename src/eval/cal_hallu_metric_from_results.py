import json
import os
from glob import glob
from tqdm import tqdm
from src.utils.util import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, auc

# uncertainty_type = "sample"  # "logits", "sample"
# tau = 0.5


def get_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for vision-language transformer"
    )

    parser.add_argument(
        "--result_type",
        type=str,
        default="combined",
        required=True,
        choices=["combined", "seperate"],
        help="Type of results to process (e.g., 'combined', 'seperate')"
    )

    parser.add_argument(
        "--image_root_dir",
        type=str,
        required=True,
        help="Root directory of images"
    )

    parser.add_argument(
        "--sup_root_dir",
        type=str,
        required=True,
        help="Path to pretrained model"
    )

    parser.add_argument(
        "--con_root_dir",
        type=str,
        help="Root directory of images"
    )

    parser.add_argument(
        "--hallu_data_dir",
        type=str,
        help="Directory containing test json files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )

    parser.add_argument(
        "--uncertainty_type",
        type=str,
        default="logits",
        choices=["logits", "sample"],
        help="Uncertainty calculation dtype"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Top-p (nucleus) sampling parameter for generation"
    )

    parser.add_argument(
        "--use_counterfactual",
        action="store_true"
    )

    # 新添加no_iou参数
    parser.add_argument(
        "--no_iou",
        action="store_true"
    )

    return parser.parse_args()


def _get_bbox_from_text(text, image_path):
    image = Image.open(image_path)
    width, height = image.size
    pred_bbox = extract_bbox_from_text(text)

    try:
        pred_bbox_abs = norm2abs_bbox(
            pred_bbox[0]["bbox_2d"], width, height) if (pred_bbox is not None) and (len(pred_bbox) != 0) else None
    except Exception as e:
        print(
            f"Error processing image {image_path} with text: {pred_bbox}. Error: {e}")
        pred_bbox_abs = None

    return pred_bbox_abs


def cal_hallu_metric_from_combined_results(results_dir, image_root_dir, out_dir, uncertainty_type="sample", tau=0.5, use_counterfactual=True):
    os.makedirs(out_dir, exist_ok=True)
    result = load_json(results_dir)

    hallu_results = []

    for item in tqdm(result):
        if uncertainty_type == 'logits':
            sup_uncertainty = item.get("sup_logits_confidence")
            con_uncertainty = item.get("con_logits_confidence")
        elif uncertainty_type == 'sample':
            sup_uncertainty = item.get("sup_sample_confidence")
            con_uncertainty = item.get("con_sample_confidence")

        sup_pred_bbox_abs = _get_bbox_from_text(
            item.get("sup_output", ""), os.path.join(image_root_dir, item["image"]))
        con_pred_bbox_abs = _get_bbox_from_text(
            item.get("con_output", ""), os.path.join(image_root_dir, item["image"]))

        if use_counterfactual:
            uncertainty, iou = cal_uncertainty_score(
                sup_pred_bbox_abs, sup_uncertainty, con_pred_bbox_abs, con_uncertainty)
        else:
            uncertainty = sup_uncertainty if sup_uncertainty is not None else 0.0
            iou = None

        if min(1 - uncertainty, 1.0) > tau:
            pred_is_hallu = True
        else:
            pred_is_hallu = False

        item['pred_is_hallu'] = pred_is_hallu
        item['uncertainty'] = min(1 - uncertainty, 1.0)
        item['iou'] = iou

        hallu_results.append(item)

    save_json(hallu_results, os.path.join(
        out_dir, f"combined_hallu_metric_uncertainty-{uncertainty_type}_tau{tau}-CON{use_counterfactual}.json"))

    return os.path.join(
        out_dir, f"combined_hallu_metric_uncertainty-{uncertainty_type}_tau{tau}-CON{use_counterfactual}.json")


def cal_hallu_metric_from_results_full(sup_results_dir, con_results_dir, image_root_dir, hallu_data_dir, out_dir, uncertainty_type="sample", tau=0.5, use_counterfactual=False, no_iou=False):
    os.makedirs(out_dir, exist_ok=True)
    sup_results_files = glob(os.path.join(sup_results_dir, "*.json"))
    con_results_files = glob(os.path.join(con_results_dir, "*.json"))

    for sup_file, con_file in zip(sup_results_files, con_results_files):
        save_file_name = os.path.basename(sup_file).replace(
            "_hf_results.json", "_hallu_metric.json")
        hall_file_name = os.path.basename(sup_file).replace(
            "_hf_results.json", "_qwen_format.json")
        hall_file_path = os.path.join(hallu_data_dir, hall_file_name)
        save_file_path = os.path.join(out_dir, save_file_name)
        sup_result = load_json(sup_file)
        con_result = load_json(con_file)
        hall_json = load_json(hall_file_path)
        hallu_data = hall_json["data"]

        sup_dict = {
            (item['image'], item['class']): item for item in sup_result
        }

        con_dict = {
            (item['image'], item['class']): item for item in con_result
        }

        matched_data = []

        for sample in hallu_data:
            key = (sample["image"], sample["class"])

            sup_item = sup_dict.get(key)
            con_item = con_dict.get(key)

            if sup_item is not None and con_item is not None:
                matched_data.append({
                    "hallu": sample,
                    "support": sup_item,
                    "contradict": con_item
                })

        hallu_results = []

        for idx, item in enumerate(matched_data):
            is_hallu = item['hallu']['is_hallu']

            if is_hallu:
                sup_res = item['contradict']
                con_res = item['support']
            else:
                sup_res = item['support']
                con_res = item['contradict']

            if uncertainty_type == 'logits':
                sup_uncertainty = sup_res.get("logits_uncertainty")
                con_uncertainty = con_res.get("logits_uncertainty")
            elif uncertainty_type == 'sample':
                sup_uncertainty = sup_res.get("sample_uncertainty")
                con_uncertainty = con_res.get("sample_uncertainty")

            sup_pred_bbox_abs = _get_bbox_from_text(
                sup_res.get("prediction", ""), os.path.join(image_root_dir, sup_res["image"]))
            con_pred_bbox_abs = _get_bbox_from_text(
                con_res.get("prediction", ""), os.path.join(image_root_dir, con_res["image"]))

            if use_counterfactual and not no_iou:
                uncertainty, iou = cal_uncertainty_score(
                    sup_pred_bbox_abs, sup_uncertainty, con_pred_bbox_abs, con_uncertainty)
            # 增加使用反事实但不使用iou的分支
            elif use_counterfactual and no_iou:
                uncertainty, iou = cal_uncertainty_score_no_iou(
                    sup_pred_bbox_abs, sup_uncertainty, con_pred_bbox_abs, con_uncertainty)
            else:
                uncertainty = sup_uncertainty if sup_uncertainty is not None else 0.0
                iou = None

            if min(1 - uncertainty, 1) > tau:
                pred_is_hallu = True
            else:
                pred_is_hallu = False

            res = {
                "idx": idx,
                "image": item['hallu']['image'],
                "class": item['hallu']['class'],
                "new_class": item['hallu']['new_class'],
                "is_hallu": is_hallu,
                "pred_is_hallu": pred_is_hallu,
                "uncertainty": min(1 - uncertainty, 1),
                "sup_pred": sup_res.get("prediction", None),
                "con_pred": con_res.get("prediction", None),
                "sup_pred_bbox_abs": sup_pred_bbox_abs,
                "con_pred_bbox_abs": con_pred_bbox_abs,
                "sup_uncertainty": sup_uncertainty,
                "con_uncertainty": con_uncertainty,
                "iou": iou
            }

            hallu_results.append(res)

        save_json(hallu_results, save_file_path)


def merge_results(out_dir):
    all_files = glob(os.path.join(out_dir, "*.json"))
    merged_results = []

    for file in all_files:
        res = load_json(file)
        merged_results.extend(res)

    return merged_results


def cal_hallu_rate(merged_results):
    y_true = [res["is_hallu"] for res in merged_results]
    y_pred = [res["pred_is_hallu"] for res in merged_results]
    y_score = [res["uncertainty"] for res in merged_results]

    cm = confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = cm.ravel()

    hallu_rate = TP/(TP+FN) if (TP+FN) > 0 else None
    precision = TP/(TP+FP) if (TP+FP) > 0 else None
    Specificity = TN/(TN+FP) if (TN+FP) > 0 else None
    acc = (TP+TN)/(TP+TN+FP+FN)
    F1 = 2*TP/(2*TP+FP+FN)

    # ====== AUC ======
    auc_score = roc_auc_score(y_true, y_score)

    print(cm)

    metrics = {
        "confusion_matrix": cm.tolist(),
        "hallu_rate": hallu_rate,
        "accuracy": acc,
        "precision": precision,
        "specificity": Specificity,
        "f1_score": F1,
        "AUC": auc_score
    }

    return metrics


if __name__ == "__main__":
    args = get_args()
    assert args.result_type in [
        'combined', 'seperate'], "Invalid result_type. Must be 'combined' or 'seperate'."
    if args.result_type == 'seperate':
        cal_hallu_metric_from_results_full(
            args.sup_root_dir, args.con_root_dir, args.image_root_dir, args.hallu_data_dir, args.output_dir, args.uncertainty_type, args.tau, args.use_counterfactual, args.no_iou)
        merged_results = merge_results(args.output_dir)
        metrics = cal_hallu_rate(merged_results)
        os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)
        save_json(metrics, os.path.join(
            args.output_dir, "metrics", "metric.json"))
    elif args.result_type == 'combined':
        cal_hallu_metric_from_combined_results(
            args.sup_root_dir, args.image_root_dir, args.output_dir, args.uncertainty_type, args.tau, args.use_counterfactual)
        metrics = cal_hallu_rate(load_json(os.path.join(
            args.output_dir, f"combined_hallu_metric_uncertainty-{args.uncertainty_type}_tau{args.tau}-CON{args.use_counterfactual}.json")))
        save_json(metrics, os.path.join(
            args.output_dir, f"metric_uncertainty-{args.uncertainty_type}_tau{args.tau}-CON{args.use_counterfactual}.json"))
