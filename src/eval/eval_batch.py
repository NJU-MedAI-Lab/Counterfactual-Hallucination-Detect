import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from glob import glob
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from src.utils.util import *
import argparse
from src.data.sys_message import SYS_MESSAGE


DEFAULT_MODALITY_MAPPING = os.getenv(
    "MODALITY_MAPPING_FILE", "YOUR_MODALITY_MAPPING_FILE"
)
modality_mapping = load_json(DEFAULT_MODALITY_MAPPING) if os.path.exists(
    DEFAULT_MODALITY_MAPPING
) else {}


def get_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for vision-language transformer"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        required=True,
        help="Root directory of images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing test json files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--test_contradictory",
        action="store_true",
        help="Whether test counterfactual query"
    )
    parser.add_argument(
        "--sys_enable",
        action="store_true",
        help="If use system prompt"
    )
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        default="none",
        choices=["logits", "sample", "all", "none"],
        help="Uncertainty calculation dtype"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=10,
        help="Number of samples for uncertainty calculation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling during generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter for generation"
    )

    return parser.parse_args()


def eval_batch_transformer(
    model,
    processor,
    image_root_dir,
    data_path,
    output_path,
    batch_size=8,
    uncertainty_type="logits",
    sample_num=10,
    temperature=0.7,
    top_p=0.9,
    test_contradictory=False,
    sys_enable=False,
):
    raw_data = load_json(data_path)

    messages_list = []
    gts_list = []
    class_list = []
    image_list = []
    ori_bbox_list = []
    norm_bbox_list = []

    modality = modality_mapping.get(raw_data["modality"]["0"], "Unknown")

    if test_contradictory:
        conversation_type = "contradictory_conversations"
    else:
        conversation_type = "supporting_conversations"

    for item in raw_data["data"]:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(image_root_dir, item["image"]),
                    },
                    {
                        "type": "text",
                        "text": item[conversation_type][0]["value"]
                        .replace("<image>\n", "")
                        .strip(),
                    },
                ],
            }
        ]
        if sys_enable:
            message = SYS_MESSAGE+message
        messages_list.append(message)
        gts_list.append(item[conversation_type][1]["value"])
        class_list.append(item["class"])
        image_list.append(item["image"])
        ori_bbox_list.append(item['bbox'])
        norm_bbox_list.append(item['norm_bbox'])

    dataset_name = raw_data["name"]
    model_name = model.config._name_or_path.split("/")[-1]

    os.makedirs(output_path, exist_ok=True)
    output_json_dir = os.path.join(
        output_path, f"{dataset_name}_{model_name}_results.json"
    )

    output_json = []

    num_samples = len(messages_list)
    processor.tokenizer.padding_side = "left"

    # 🔥 推理阶段：关闭梯度
    with torch.inference_mode():
        for start in tqdm(range(0, num_samples, batch_size), desc=dataset_name):
            end = min(start + batch_size, num_samples)

            batch_messages = messages_list[start:end]
            batch_gts = gts_list[start:end]
            batch_classes = class_list[start:end]
            batch_images = image_list[start:end]
            batch_bbox = ori_bbox_list[start:end]
            batch_norm_bbox = norm_bbox_list[start:end]

            inputs = processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

            generated_ids = outputs.sequences
            scores = outputs.scores
            logits = torch.stack(scores, dim=1)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids, generated_ids
                )
            ]

            if uncertainty_type == "logits":
                logits_uncertainties = cal_logits_uncertainty(
                    generated_ids_trimmed, logits)
            elif uncertainty_type == "sample":
                sample_uncertainties = cal_sample_uncertainty(
                    model, processor, inputs, batch_images, image_root_dir, sample_num, temperature, top_p)
            elif uncertainty_type == "all":
                logits_uncertainties = cal_logits_uncertainty(
                    generated_ids_trimmed, logits)
                sample_uncertainties = cal_sample_uncertainty(
                    model, processor, inputs, batch_images, image_root_dir, sample_num, temperature, top_p)

            batch_outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, output_text in enumerate(batch_outputs):
                idx = start + i
                if uncertainty_type == "logits":
                    logits_uncertainty = logits_uncertainties[i]
                    output_json.append(
                        {
                            "idx": idx,
                            "image": batch_images[i],
                            "modality": modality,
                            "class": batch_classes[i],
                            "ground_truth": batch_gts[i],
                            "prediction": output_text,
                            "logits_uncertainty": logits_uncertainty,
                            "bbox": batch_bbox[i],
                            "norm_bbox": batch_norm_bbox[i]
                        }
                    )
                elif uncertainty_type == "sample":
                    sample_uncertainty = sample_uncertainties[i]["stability_confidence"]
                    sample_bboxes = sample_uncertainties[i]["sample_bboxes"]
                    output_json.append(
                        {
                            "idx": idx,
                            "image": batch_images[i],
                            "modality": modality,
                            "class": batch_classes[i],
                            "ground_truth": batch_gts[i],
                            "prediction": output_text,
                            "sample_uncertainty": sample_uncertainty,
                            "bbox": batch_bbox[i],
                            "norm_bbox": batch_norm_bbox[i],
                            "sample_bboxes": sample_bboxes
                        }
                    )
                elif uncertainty_type == "all":
                    logits_uncertainty = logits_uncertainties[i]
                    sample_uncertainty = sample_uncertainties[i]["stability_confidence"]
                    sample_bboxes = sample_uncertainties[i]["sample_bboxes"]
                    output_json.append(
                        {
                            "idx": idx,
                            "image": batch_images[i],
                            "modality": modality,
                            "class": batch_classes[i],
                            "ground_truth": batch_gts[i],
                            "prediction": output_text,
                            "logits_uncertainty": logits_uncertainty,
                            "sample_uncertainty": sample_uncertainty,
                            "bbox": batch_bbox[i],
                            "norm_bbox": batch_norm_bbox[i],
                            "sample_bboxes": sample_bboxes
                        }
                    )
                else:
                    output_json.append(
                        {
                            "idx": idx,
                            "image": batch_images[i],
                            "modality": modality,
                            "class": batch_classes[i],
                            "ground_truth": batch_gts[i],
                            "prediction": output_text,
                            "bbox": batch_bbox[i],
                            "norm_bbox": batch_norm_bbox[i]
                        }
                    )

            # 🔥 显存释放（与 single 对齐）
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

    save_json(output_json, output_json_dir)


def load_model_with_lora_merge(
    model_path,
    torch_dtype,
    attn_implementation="flash_attention_2",
    device_map="auto",
):
    """
    如果 model_path 是 LoRA adapter：
        1. 读取 adapter_config.json
        2. 自动加载 base model
        3. merge_and_unload
    否则：
        直接当普通模型加载
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print(
            f"[INFO] Detected LoRA adapter at {model_path}, merging weights...")

        # 1️⃣ 先读 adapter config，拿 base_model_name_or_path
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        print(f"[INFO] Base model: {base_model_path}")

        # 2️⃣ 加载 base model
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        # 3️⃣ 加载 LoRA 并 merge
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()

        print("[INFO] LoRA merged successfully.")

    else:
        print(f"[INFO] Loading normal model from {model_path}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        # for name, _ in model.named_parameters():
        #     if "lora" in name.lower():
        #         print("Found LoRA param:", name)
        #     else:
        #         print("Found non-LoRA param:", name)

    return model


def main():
    args = get_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # ✅ 模型只加载一次,判断是否是lora model
    model = load_model_with_lora_merge(
        model_path=args.model_path,
        torch_dtype=dtype_map[args.dtype],
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, fix_mistral_regex=True)

    data_paths = sorted(glob(os.path.join(args.data_dir, "*.json")))
    already_done = sorted(glob(os.path.join(
        args.output_path, "*_results.json")))

    for data_path in data_paths:
        dataset_name = data_path.split("/")[-1].split("_qwen_format.json")[0]
        output_json_dir = os.path.join(
            args.output_path, f"{dataset_name}_{model.config._name_or_path.split('/')[-1]}_results.json"
        )
        if output_json_dir in already_done:
            print(f"Skipping {data_path} as results already exist.")
            continue

        print(f"Testing {data_path}...")
        eval_batch_transformer(
            model=model,
            processor=processor,
            image_root_dir=args.image_root_path,
            data_path=data_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            uncertainty_type=args.uncertainty_type,
            sample_num=args.sample_num,
            temperature=args.temperature,
            top_p=args.top_p,
            test_contradictory=args.test_contradictory,
            sys_enable=args.sys_enable,
        )

    # 🔥 最终释放
    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
