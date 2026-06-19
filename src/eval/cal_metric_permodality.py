import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from glob import glob
from ..utils.util import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate hallucination metrics per modality")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the combined results JSON file')
    return parser.parse_args()


args = get_args()

results_dir = args.results_dir
data_dir = os.getenv("HALLU_TEST_DATA_DIR", "YOUR_HALLU_TEST_DATA_DIR")
modality_mapping = load_json(os.getenv(
    "MODALITY_MAPPING_FILE", "YOUR_MODALITY_MAPPING_FILE"))

result = load_json(results_dir)

total_modalities = []
models = ['gemini-3-flash-preview',
          'gpt-5.1-2025-11-13', 'grok-4-fast', 'medgemma-27b']

for item in result:
    dataset_name = item['dataset']
    data_path = os.path.join(data_dir, dataset_name+'_qwen_format.json')
    ori_data = load_json(data_path)
    modality = modality_mapping.get(ori_data['modality']['0'], 'Unknown')
    if modality != 'CT' and modality != 'MRI':
        modality = 'Other'
    item['modality'] = modality
    if modality not in total_modalities:
        total_modalities.append(modality)

metric_results = {}
for model in models:
    metric_results[model] = {}
    for modality in total_modalities:
        print(f"Evaluating {model} on {modality}:")
        y_true = []
        y_pred = []
        y_score = []
        for item in result:
            if item['model'] == model and item['modality'] == modality:
                y_true.append(item['is_hallu'])
                y_pred.append(item['pred_is_hallu'])
                y_score.append(item['uncertainty'])
        if y_true and y_pred:
            cm = confusion_matrix(y_true, y_pred)

            TN, FP, FN, TP = cm.ravel()

            hallu_rate = TP/(TP+FN) if (TP+FN) > 0 else None
            precision = TP/(TP+FP) if (TP+FP) > 0 else None
            Specificity = TN/(TN+FP) if (TN+FP) > 0 else None
            acc = (TP+TN)/(TP+TN+FP+FN)
            F1 = 2*TP/(2*TP+FP+FN)

            # ====== AUC ======
            auc_score = roc_auc_score(y_true, y_score)

            metrics = {
                "confusion_matrix": cm.tolist(),
                "hallu_rate": hallu_rate,
                "accuracy": acc,
                "precision": precision,
                "specificity": Specificity,
                "f1_score": F1,
                "auc": auc_score
            }
            metric_results[model][modality] = metrics
            print(
                f"Confusion Matrix for {model} on {modality}:\n{cm}")
        else:
            print(f"No data for {model} on {modality}.")
    print("\n" + "="*50 + "\n")

output_dir = results_dir.replace('.json', '_per_model_metrics.json')
with open(output_dir, 'w') as f:
    json.dump(metric_results, f, indent=4)
