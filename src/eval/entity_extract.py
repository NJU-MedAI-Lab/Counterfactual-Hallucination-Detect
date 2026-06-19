import random
import time
from openai import OpenAI
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
from glob import glob
from src.utils.util import *
import numpy as np
import torch
from openai import OpenAI
import re


def clean_json_output(content):
    content = content.strip()
    # 去掉 ```json 和 ```
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content


def extract_entity_from_report(
    client,
    report,
    entity_list,
    model="gemini-2.5-flash",
    max_retries=5,
    base_delay=1.0,
    api_config=None,
):
    """
    Extract a single medical entity from report under entity list constraint,
    with retry logic.
    """

    system_prompt = (
        "You are a medical AI assistant specialized in radiology and clinical report understanding.\n"
        "Your task is to identify the single most relevant medical entity from a given candidate entity list,\n"
        "based on the medical report text.\n"
        "You must strictly choose ONE entity from the provided entity list.\n"
        "You must NOT invent new entities."
    )

    user_prompt = f"""
        You are given a medical report.

        Your task is to identify the SINGLE most relevant medical entity
        from the provided candidate entity list, based ONLY on the medical report text.

        The target entity may belong to ONE of two types:
        1. Lesion / abnormal finding(positive only)
        2. Anatomical organ(normal state)

        You MUST follow the decision rules below.

        ------------------------------------------------
        Decision Rules
        ------------------------------------------------

        Step 1 — Check for Positive Lesion Presence

        You MUST FIRST determine whether the report mentions any POSITIVE lesion or abnormal finding.

        A positive lesion includes explicit abnormalities such as:
        mass, tumor, lesion, edema, nodule, fracture, effusion, infarction, consolidation, etc.

        • If the report mentions a POSITIVE lesion, you MUST select a lesion entity from the Lesion Candidate List.
        • If the report does NOT mention any positive lesion(including statements like "no lesion", "without abnormality", or no mention at all), you MUST select an organ entity from the Organ Candidate List.

        ------------------------------------------------
        Step 2 — Entity Selection Rules

        General rules(apply to BOTH types):

        1. You MUST select exactly ONE entity.
        2. You MUST NOT generate, modify, or paraphrase any entity.
        3. The output entity MUST match one item in the provided candidate lists EXACTLY.
        4. If multiple entities seem relevant, choose the MOST specific one.
        5. Use ONLY information from the medical report.

        ------------------------------------------------
        Additional Rules for Lesion Entity

        If the selected entity is a lesion:

        • The lesion MUST be selected from the Lesion Candidate List.
        • The entity SHOULD describe both the lesion type and its anatomical location in a phrase like "a tumor in the kidney".
        • Choose the candidate that best matches the lesion and organ explicitly mentioned in the report.

        ------------------------------------------------
        Additional Rules for Organ Entity

        If the selected entity is an anatomical organ:

        • The organ MUST be selected from the Organ Candidate List.
        • The entity MUST include the prefix "a normal" (e.g., "a normal lung").
        • Output ONLY the exact organ entity with the "a normal" prefix as it appears in the candidate list.

        ------------------------------------------------
        Output Requirements

        • Output JSON ONLY.
        • Do NOT output explanations.
        • Do NOT output extra text.
        • Entity names MUST be copied EXACTLY from the candidate list.

        ------------------------------------------------
        Output Format

        If the selected entity is an ORGAN:

        {{
            "entity_type": "organ",
            "entity": "<one entity from candidate list>"
        }}

        If the selected entity is a LESION:

        {{
            "entity_type": "lesion",
            "entity": "<lesion entity from candidate list>"
        }}

        ------------------------------------------------
        [Medical Report]
        {report}

        [Lesion Candidate List]
        {entity_list['all_lesion_classes']}

        [Organ Candidate List]
        {entity_list['all_organ_classes']}

        ------------------------------------------------
        Example 1 — Organ

        Medical Report:
        "The left lung shows no focal consolidation or pleural effusion."

        Candidate Entity List:
        ["a normal left lung", "a normal pleural effusion", "a normal heart"]

        Correct Output:
        {{
            "entity_type": "organ",
            "entity": "a normal left lung"
        }}

        ------------------------------------------------
        Example 2 — Lesion

        Medical Report:
        "There is a pulmonary nodule in the right lung."

        Candidate Entity List:
        ["a pulmonary nodule in the right lung",
            "a normal right lung", "a normal heart"]

        Correct Output:
        {{
            "entity_type": "lesion",
            "entity": "a pulmonary nodule in the right lung"
        }}

        ------------------------------------------------
        Example 3 — Organ(No Lesion Mentioned)

        Medical Report:
        "There is no edema in the brain."

        Candidate Entity List:
        ["a normal brain", "no edema", "a tumor in the brain"]

        Correct Output:
        {{
            "entity_type": "organ",
            "entity": "a normal brain"
        }}
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    },
                ],
                temperature=api_config.temperature if api_config else 0.1,
                max_tokens=api_config.max_tokens if api_config else 64,
            )

            content = response.choices[0].message.content
            content = clean_json_output(content)
            result = json.loads(content)

            # -------- 关键：结果合法性校验 --------
            if "entity" not in result:
                raise ValueError("Missing 'entity' field in JSON output")

            if result["entity"] not in entity_list['all_classes'] and attempt != max_retries:
                raise ValueError(
                    f"Entity '{result['entity']}' not in candidate list"
                )

            return result  # ✅ 成功直接返回

        except Exception as e:
            last_error = e

            if attempt == max_retries:
                break

            # 指数退避 + jitter
            sleep_time = base_delay * (2 ** (attempt - 1))
            sleep_time += random.uniform(0, 0.2)

            print(
                f"[Retry {attempt}/{max_retries}] "
                f"Failed with error: {e}. Retrying in {sleep_time:.2f}s..."
            )

            time.sleep(sleep_time)

    # 所有重试失败
    raise RuntimeError(
        f"Failed after {max_retries} retries. Last error:\n{last_error}"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract entities from reports.")
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--entity_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--base_url", type=str,
                        default=os.getenv("OPENAI_BASE_URL", "YOUR_LLM_BASE_URL"))
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("OPENAI_API_KEY", "YOUR_LLM_API_KEY"))
    parser.add_argument("--max_retries", type=int, default=4)
    args = parser.parse_args()

    # Configuration
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    MODEL_NAME = args.model
    MAX_RETRIES = args.max_retries

    reports = load_json(args.report_json)
    entity_list = load_json(args.entity_file)
    entity_list = entity_list

    for item in tqdm(reports):
        report = item["report"]
        report = report.lower()
        report = report.split("**findings:**")[-1].strip()
        item["entity_extract_model"] = MODEL_NAME
        item["norm_report"] = report
        extracted_entity = extract_entity_from_report(
            client, report, entity_list, model=MODEL_NAME, max_retries=MAX_RETRIES
        )
        item["extracted_entity"] = extracted_entity["entity"]
        item["extracted_entity_type"] = extracted_entity["entity_type"]

        save_json(reports, args.output_file)


if __name__ == "__main__":
    main()
