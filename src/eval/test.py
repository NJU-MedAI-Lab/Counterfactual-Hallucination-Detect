import warnings
from pathlib import Path
import sys
from openai import OpenAI
import json
import time
import os
import re
import random
from typing import List
import difflib
import transformers
from transformers import AutoProcessor, Trainer, AutoConfig
from src.train.argument import (
    ModelArguments,
    DataArguments,
    TestArguments,
    TrainingArguments,
)
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from src.data.test_data_processor import _do_entity_extraction, _get_dataset, _create_counterfactuals, TestDataBatch
from tqdm import tqdm
from src.utils.util import *
from src.data.sys_message import SYS_MESSAGE
import torch.distributed as dist
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _build_message(image_root_dir, item, conversation_name, sys_enable=False):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(image_root_dir, item['image'])
                },
                {
                    "type": "text",
                    "text": item[conversation_name][0]["value"]
                    .replace("<image>\n", " ").strip(),
                }
            ]
        }
    ]
    if sys_enable:
        return SYS_MESSAGE + message
    else:
        return message


def get_batch_data(list_data_dict, test_args, data_args):
    sup_messages_list = []
    con_messages_list = []
    sup_entity_list = []
    con_entity_list = []
    is_hallu_list = []
    class_list = []
    gt_sup_list = []
    gt_con_list = []
    ori_bbox_list = []
    norm_bbox_list = []
    model_list = []
    image_list = []
    dataset_name_list = []
    for item in list_data_dict:
        if test_args.do_entity_extract:
            sup_message = _build_message(
                item['data_path'], item, conversation_name="extracted_entity_sup_conversations", sys_enable=data_args.sys_enable)
            con_message = _build_message(
                item['data_path'], item, conversation_name="extracted_entity_con_conversations", sys_enable=data_args.sys_enable)
            sup_entity = item["extracted_entity_sup"]
            con_entity = item["extracted_entity_con"]
            sup_messages_list.append(sup_message)
            con_messages_list.append(con_message)
            sup_entity_list.append(sup_entity)
            con_entity_list.append(con_entity)
        else:
            sup_message = _build_message(
                item['data_path'], item, conversation_name="supporting_conversations", sys_enable=data_args.sys_enable)
            con_message = _build_message(
                item['data_path'], item, conversation_name="contradictory_conversations", sys_enable=data_args.sys_enable)
            sup_entity = item["sup_phrase"]
            con_entity = item["con_phrase"]
            if item["is_hallu"]:
                sup_messages_list.append(con_message)
                con_messages_list.append(sup_message)
                sup_entity_list.append(con_entity)
                con_entity_list.append(sup_entity)
            else:
                sup_messages_list.append(sup_message)
                con_messages_list.append(con_message)
                sup_entity_list.append(sup_entity)
                con_entity_list.append(con_entity)

        is_hallu_list.append(item["is_hallu"])
        class_list.append(item["new_class"])
        gt_sup_list.append(item["sup_phrase"]) if not item["is_hallu"] else gt_sup_list.append(
            item["con_phrase"])
        gt_con_list.append(item["con_phrase"]) if not item["is_hallu"] else gt_con_list.append(
            item["sup_phrase"])
        ori_bbox_list.append(item["bbox"])
        norm_bbox_list.append(item["norm_bbox"])
        model_list.append(item["model"])
        image_list.append(item['image'])
        dataset_name_list.append(item['dataset'])

    return TestDataBatch(
        sup_messages=sup_messages_list,
        con_messages=con_messages_list,
        sup_entities=sup_entity_list,
        con_entities=con_entity_list,
        is_hallu=is_hallu_list,
        class_list=class_list,
        gt_sup=gt_sup_list,
        gt_con=gt_con_list,
        ori_bbox=ori_bbox_list,
        norm_bbox=norm_bbox_list,
        model_list=model_list,
        image_list=image_list,
        dataset_name_list=dataset_name_list,
    )


def main(attn_implementation='flash_attention_2'):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TestArguments, TrainingArguments))
    model_args, data_args, test_args, training_args = parser.parse_args_into_dataclasses()

    if not test_args.test_only:
        raise ValueError("Please set --test_only for test.py")

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif 'qwen3' in model_args.model_name_or_path.lower():
        config = Qwen3VLConfig.from_pretrained(
            model_args.model_name_or_path,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            device_map='auto',
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(
        f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        fix_mistral_regex=True
    )

    processor.tokenizer.padding_side = "left"  # set left padding for generation

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if test_args.do_entity_extract:
        print("Performing entity extraction on test set...")
        list_data_dict = _do_entity_extraction(data_args, test_args)
        with open(os.path.join(training_args.output_dir, "extracted_entity_results.json"), "w") as f:
            json.dump(list_data_dict, f, indent=4)

    else:
        list_data_dict = _get_dataset(data_args)

    if test_args.do_counterfactual_test:
        print("Performing counterfactual generation...")

        list_data_dict = _create_counterfactuals(
            list_data_dict, test_args.do_entity_extract, test_args)
        with open(os.path.join(training_args.output_dir, "counterfactual_data.json"), "w") as f:
            json.dump(list_data_dict, f, indent=4)

    batch_data = get_batch_data(list_data_dict, test_args, data_args)

    model.eval()  # set eval mode

    print("Start testing...")

    test(batch_data, model, processor, tokenizer, training_args, test_args)

    if dist.is_initialized():
        dist.destroy_process_group()


def test(batch_data, model, processor, tokenizer, training_args, test_args):
    # Prepare output file
    output_path = os.path.join(
        training_args.output_dir, test_args.test_result_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_samples = len(batch_data.sup_messages)

    output_json = []

    with torch.inference_mode():
        for start in tqdm(range(0, num_samples, test_args.test_batch_size), desc="Testing"):
            end = min(start + test_args.test_batch_size, num_samples)
            batch_sup_messages = batch_data.sup_messages[start:end]
            batch_con_messages = batch_data.con_messages[start:end]
            batch_sup_entities = batch_data.sup_entities[start:end]
            batch_con_entities = batch_data.con_entities[start:end]
            batch_is_hallu = batch_data.is_hallu[start:end]
            batch_class_list = batch_data.class_list[start:end]
            batch_gt_sup = batch_data.gt_sup[start:end]
            batch_gt_con = batch_data.gt_con[start:end]
            batch_ori_bbox = batch_data.ori_bbox[start:end]
            batch_norm_bbox = batch_data.norm_bbox[start:end]
            batch_model_list = batch_data.model_list[start:end]
            batch_images = batch_data.image_list[start:end]
            batch_dataset_name_list = batch_data.dataset_name_list[start:end]

            sup_inputs = processor.apply_chat_template(
                batch_sup_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
            ).to(model.device)

            sup_outputs = model.generate(
                **sup_inputs,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = sup_outputs.sequences
            scores = sup_outputs.scores
            # (batch_size, seq_len, vocab_size)
            logits = torch.stack(scores, dim=1)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    sup_inputs.input_ids, generated_ids
                )
            ]

            if test_args.uncertainty_type == "logits":
                sup_logits_uncertainties = cal_logits_uncertainty(
                    generated_ids_trimmed, logits)
                sup_sample_uncertainties = None
            elif test_args.uncertainty_type == "sample":
                sup_logits_uncertainties = None
                sup_sample_uncertainties = cal_sample_uncertainty(
                    model, processor, sup_inputs, batch_images, test_args.image_root_dir, test_args.sample_num, test_args.temperature, test_args.top_p)
            elif test_args.uncertainty_type == "all":
                sup_logits_uncertainties = cal_logits_uncertainty(
                    generated_ids_trimmed, logits)
                sup_sample_uncertainties = cal_sample_uncertainty(
                    model, processor, sup_inputs, batch_images, test_args.image_root_dir, test_args.sample_num, test_args.temperature, test_args.top_p)

            sup_batch_outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if test_args.do_counterfactual_test:
                con_inputs = processor.apply_chat_template(
                    batch_con_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                ).to(model.device)

                con_outputs = model.generate(
                    **con_inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                generated_ids = con_outputs.sequences
                scores = con_outputs.scores
                # (batch_size, seq_len, vocab_size)
                logits = torch.stack(scores, dim=1)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(
                        con_inputs.input_ids, generated_ids
                    )
                ]

                if test_args.uncertainty_type == "logits":
                    con_logits_uncertainties = cal_logits_uncertainty(
                        generated_ids_trimmed, logits)
                    con_sample_uncertainties = None
                elif test_args.uncertainty_type == "sample":
                    con_logits_uncertainties = None
                    con_sample_uncertainties = cal_sample_uncertainty(
                        model, processor, con_inputs, batch_images, test_args.image_root_dir, test_args.sample_num, test_args.temperature, test_args.top_p)
                elif test_args.uncertainty_type == "all":
                    con_logits_uncertainties = cal_logits_uncertainty(
                        generated_ids_trimmed, logits)
                    con_sample_uncertainties = cal_sample_uncertainty(
                        model, processor, con_inputs, batch_images, test_args.image_root_dir, test_args.sample_num, test_args.temperature, test_args.top_p)

                con_batch_outputs = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            if test_args.do_counterfactual_test:
                for i, sup_output in enumerate(sup_batch_outputs):
                    output_json.append({
                        "idx": start + i,
                        "dataset": batch_dataset_name_list[i],
                        "image": batch_images[i],
                        "new_class": batch_class_list[i],
                        "model": batch_model_list[i],
                        "is_hallu": batch_is_hallu[i],
                        "do_entity_extract": test_args.do_entity_extract,
                        "do_counterfactual_test": test_args.do_counterfactual_test,
                        "confidence_type": test_args.uncertainty_type,
                        "sup_logits_confidence": sup_logits_uncertainties[i] if sup_logits_uncertainties is not None else None,
                        "con_logits_confidence": con_logits_uncertainties[i] if con_logits_uncertainties is not None else None,
                        "sup_sample_confidence": sup_sample_uncertainties[i]["stability_confidence"] if sup_sample_uncertainties is not None else None,
                        "con_sample_confidence": con_sample_uncertainties[i]["stability_confidence"] if con_sample_uncertainties is not None else None,
                        "gt_sup": batch_gt_sup[i],
                        "gt_con": batch_gt_con[i],
                        "sup_entity": batch_sup_entities[i],
                        "con_entity": batch_con_entities[i],
                        "sup_output": sup_output,
                        "con_output": con_batch_outputs[i],
                        "ori_bbox": batch_ori_bbox[i],
                        "norm_bbox": batch_norm_bbox[i],
                        "sup_sample_bboxes": sup_sample_uncertainties[i]["sample_bboxes"] if sup_sample_uncertainties is not None else None,
                        "con_sample_bboxes": con_sample_uncertainties[i]["sample_bboxes"] if con_sample_uncertainties is not None else None,
                    })

            else:
                for i, sup_output in enumerate(sup_batch_outputs):
                    output_json.append({
                        "idx": start + i,
                        "dataset": batch_dataset_name_list[i],
                        "image": batch_images[i],
                        "model": batch_model_list[i],
                        "is_hallu": batch_is_hallu[i],
                        "do_entity_extract": test_args.do_entity_extract,
                        "do_counterfactual_test": test_args.do_counterfactual_test,
                        "confidence_type": test_args.uncertainty_type,
                        "sup_logits_confidence": sup_logits_uncertainties[i] if sup_logits_uncertainties is not None else None,
                        "sup_sample_confidence": sup_sample_uncertainties[i]["stability_confidence"] if sup_sample_uncertainties is not None else None,
                        "gt_sup": batch_gt_sup[i],
                        "sup_entity": batch_sup_entities[i],
                        "sup_output": sup_output,
                        "ori_bbox": batch_ori_bbox[i],
                        "norm_bbox": batch_norm_bbox[i],
                        "sup_sample_bboxes": sup_sample_uncertainties[i]["sample_bboxes"] if sup_sample_uncertainties is not None else None,
                    })

            # 🔥 显存释放（与 single 对齐）
            del sup_inputs, sup_outputs, generated_ids, generated_ids_trimmed
            if test_args.do_counterfactual_test:
                del con_inputs, con_outputs
            torch.cuda.empty_cache()

    print("Testing completed.")

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"Test results saved to {output_path}")


if __name__ == "__main__":
    main(attn_implementation='flash_attention_2')
