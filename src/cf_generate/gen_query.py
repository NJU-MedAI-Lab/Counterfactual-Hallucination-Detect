from openai import OpenAI
import os
import json
import time
import re
import random
import base64
from typing import List
import difflib
from .knowledge import PATHOLOGY_KEYWORDS, PATHOLOGICAL_STATE_WORDS, NON_OVERLAPPING_COOCURRENCE, LESION_LOCATION, LESION_NAME, ANATOMY_RULES

# ----------------------------
# Helper Functions
# ----------------------------


def is_disease(label: str) -> bool:
    """判断 label 是否为疾病，仅基于提供的病理词典。"""
    label_lower = label.lower().strip()
    return any(kw.lower() in label_lower for kw in PATHOLOGY_KEYWORDS)


def get_opposite_side(label: str) -> str:
    # Step 1: 标准化分隔符：把下划线转为空格（便于单词边界识别）
    normalized = label.replace('_', ' ')

    # Step 2: 检查是否包含 left/right 作为独立词
    if re.search(r'\b[Ll]eft\b', normalized):
        # 替换原始 label 中的 left/Left -> right
        result = re.sub(r'\b[Ll]eft\b', 'right',
                        normalized, flags=re.IGNORECASE)
        return result
    elif re.search(r'\b[Rr]ight\b', normalized):
        result = re.sub(r'\b[Rr]ight\b', 'left',
                        normalized, flags=re.IGNORECASE)
        return result
    else:
        raise ValueError(
            f"Label '{label}' is not a lateralized anatomical structure.")


def get_non_overlapping_counterpart(client, label: str, candidate_class_root: str, model: str) -> str:
    """
    从 NON_OVERLAPPING_COOCURRENCE 中选取一个不重叠器官。
    - 先尝试精确匹配（label → lowercase + space_to_underscore）
    - 若失败，使用模糊匹配（closest key by similarity）
    """
    all_classes = json.load(open(candidate_class_root, 'r'))
    label_clean = label.strip().lower()
    label_key = label_clean.replace(' ', '_')

    # Step 1: Exact match
    if label_key in NON_OVERLAPPING_COOCURRENCE:
        candidates = NON_OVERLAPPING_COOCURRENCE[label_key]
        if candidates:
            return random.choice(candidates).replace('_', ' ')

    # Step 2: Fuzzy match against all keys
    all_keys = list(NON_OVERLAPPING_COOCURRENCE.keys())
    # Normalize keys for comparison: lower + space/underscore unified
    normalized_keys = [k.lower().replace('_', ' ') for k in all_keys]
    # Find closest match based on normalized label
    closest_norm = difflib.get_close_matches(
        label_clean, normalized_keys, n=1, cutoff=0.5)

    if closest_norm:
        # Map back to original key
        idx = normalized_keys.index(closest_norm[0])
        original_key = all_keys[idx]
        candidates = NON_OVERLAPPING_COOCURRENCE[original_key]
        if candidates:
            return random.choice(candidates).replace('_', ' ')

    # Step 3: 如果模糊匹配也失败，使用大模型生成
    prompt = f"""
        You are a medical anatomy expert.

        Your task is to generate ONE anatomical organ that is guaranteed NOT to appear in the same medical image as the given target organ.

        The target organ is NOT included in any predefined non-overlapping co-occurrence rules. You must infer the counterfactual organ using general anatomical knowledge and the rules below.

        ====================
        Anatomy Knowledge
        ====================

        Lateralized indicators:
        {ANATOMY_RULES["lateralized"]}

        Paired structures:
        {ANATOMY_RULES["paired_structures"]}

        Bilateral organs:
        {ANATOMY_RULES["bilateral_organs"]}

        Midline organs:
        {ANATOMY_RULES["midline_organs"]}

        Always left-sided organs:
        {ANATOMY_RULES["always_left_structures"]}

        Always right-sided organs:
        {ANATOMY_RULES["always_right_structures"]}

        Anatomical spaces:
        {ANATOMY_RULES["spaces"]}

        ====================
        Selection Rules
        ====================

        1. Select an organ that belongs to a clearly different anatomical region from the target organ:
        - Cranial structures should select thoracic or pelvic organs.
        - Thoracic structures should select cranial or pelvic organs.
        - Abdominal structures should select cranial or thoracic organs.
        - Pelvic structures should select cranial or thoracic organs.

        2. If the target organ is left or right sided:
        - You may select the contralateral paired structure
        OR
        - Select a distant organ from a different body region.

        3. Respect fixed laterality:
        - spleen must be left-sided
        - gallbladder must be right-sided

        4. The output MUST be a pure anatomical organ label.

        Do NOT include any pathological descriptors or state words:
        {PATHOLOGICAL_STATE_WORDS}

        5. The output MUST exist in the anatomy knowledge lists and in the candidate class list:
        {all_classes['all_classes']}
        
        6. The output MUST be a single organ, not a combination of multiple organs.
        
        7. Output ONLY the organ name, in lowercase, without punctuation or extra words.

        ====================
        Output Requirement
        ====================

        Input organ: {label}

        Output ONLY one organ label.
        Do NOT output explanation.
        Do NOT output punctuation.
        Do NOT output multiple labels.
    """

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=24,
                temperature=0.15,  # 低温保证一致性
                timeout=10
            )
            phrase = response.choices[0].message.content.strip()
            phrase = re.sub(
                r'^["“”‘’\'`•\-\s]+|["“”‘’\'`•\-\s.]+$', '', phrase)
            phrase = re.sub(r'\s+', ' ', phrase).lower()
            if phrase:
                return phrase
        except Exception as e:
            print(
                f"[API Error] Attempt {attempt + 1} for '{label}': {e}")
            time.sleep(1.0)


def generate_counterfactual_for_disease(disease_label: str, location: str) -> str:
    if not location:
        raise ValueError(
            f"Location must be provided for disease label '{disease_label}'.")
    count_label = f'a normal {location}'
    return count_label


def generate_counterfactual_phrase(client, label: str, type: str, location: str, candidate_class_root: str, model: str) -> str:
    label_clean = label.strip()
    label_lower = label_clean.lower()

    # Rule 2: 检查是否为侧别解剖结构（优先处理！）
    if ('left' in label_lower or 'right' in label_lower) and type != 'LESION':
        try:
            raw_cf = get_opposite_side(label_clean)
            print(raw_cf)
            return humanize_label_via_llm(client, raw_cf, type="ORGAN", location=None, model=model)
        except ValueError:
            pass  # fallback to Rule 3

    # Rule 1: 如果不是解剖结构，再判断是否为疾病
    if type == 'LESION' or is_disease(label_clean):
        return generate_counterfactual_for_disease(label_clean, location)

    # Rule 3: 非侧别器官
    raw_cf = get_non_overlapping_counterpart(
        client, label_clean, candidate_class_root, model)
    return humanize_label_via_llm(client, raw_cf, type="ORGAN", location=None, model=model)


def humanize_label_via_llm(client, label: str, type: str, location: str, model: str) -> str:
    """
    Use LLM to convert a raw anatomical/pathological label (e.g., 'lung_left', 'pneumothorax')
    into a natural, human-readable phrase commonly used in radiology reports.
    - No manual pattern matching.
    - Output should be fluent, concise, and clinically appropriate.
    """
    if not label or not isinstance(label, str):
        return label

    prompt = (
        f"Convert the following medical label into a natural and concise radiology-style phrase.\n\n"

        f"You are given:\n"
        f"- Label\n"
        f"- Label type (ORGAN or LESION)\n"
        f"- Anatomical location (always provided for LESION)\n\n"

        f"Strict requirements:\n"
        f"------------------------------------------------\n"

        f"If label type is ORGAN:\n"
        f"- Convert the label into a natural anatomical organ name.\n"
        f"- You MUST preserve ALL anatomical details from the label, including:\n"
        f"  • laterality (left, right, bilateral, etc.)\n"
        f"  • anatomical subregions (e.g., lobe, segment, chamber, cortex, etc.)\n"
        f"  • positional descriptors appearing in the label\n"
        f"- You MUST NOT remove or simplify any anatomical words.\n"
        f"- Output format MUST be:\n"
        f"  a normal <natural anatomical organ name>\n"
        f"- Do NOT use the location field.\n"
        f"- Do NOT include pathological or diagnostic terms.\n\n"

        f"If label type is LESION:\n"
        f"- The lesion name MUST be derived from the label.\n"
        f"- The anatomical organ name MUST be EXACTLY the provided location.\n"
        f"- Output format MUST be:\n"
        f"  a <lesion name> in the <location>\n"
        f"- Do NOT modify, paraphrase, or expand the location.\n\n"

        f"Forbidden ORGAN simplification examples:\n"
        f"- 'left kidney' → 'kidney' (WRONG)\n"
        f"- 'right ventricle' → 'ventricle' (WRONG)\n"
        f"- 'left lower lobe lung' → 'lung' (WRONG)\n\n"

        f"General rules:\n"
        f"1. Preserve the original medical meaning exactly.\n"
        f"2. Do NOT add explanations, modifiers, or uncertainty words.\n"
        f"3. Output ONLY the phrase.\n"
        f"4. Use lowercase letters only.\n"
        f"5. Do NOT add ending punctuation.\n\n"

        f"Label: {label}\n"
        f"Label type: {type}\n"
        f"Location: {location}"
    )

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=1,  # very low for determinism
                timeout=8
            )
            result = response.choices[0].message.content.strip()
            # Clean: remove quotes, markdown, trailing punctuation
            result = re.sub(
                r'^["“”‘’\'`•\-\s]+|["“”‘’\'`•\-\s.,;:!?]+$', '', result)
            result = re.sub(r'\s+', ' ', result).lower()
            if result:
                return result
        except Exception as e:
            print(
                f"[Humanize LLM Error] Attempt {attempt + 1} for '{label}': {e}")
            time.sleep(0.5)

    # Fallback: return original label (lowercase, cleaned and no '_' mark)
    return label.strip().lower().replace('_', ' ')


def _generate_prompt(phrase: str) -> str:
    """智能生成 prompt，避免重复 'the'"""
    phrase_clean = phrase.strip().lower()

    # 检查是否以冠词开头
    article_start = phrase_clean.startswith(('the ', 'a ', 'an '))

    if article_start:
        return f"If there is {phrase}, please locate it in the image with its bbox coordinates and its label and output in JSON format."
    else:
        return f"If there is the {phrase}, please locate it in the image with its bbox coordinates and its label and output in JSON format."


def generate_supporting_and_counter_phrase(client, label: str, type: str, candidate_class_root: str, model: str) -> dict:
    """
    生成包含支持性与矛盾性描述的完整结果字典，用于反事实数据构建。
    Args:
        label (str): 原始标签（如 'lung_left', 'pneumothorax'）
    Returns:
        dict: 结构化结果，可直接存入 JSON 或用于训练
    """
    if not is_disease(label) or type == 'organ':
        # Step 1: Humanize the label → supporting phrase
        supporting_phrase = humanize_label_via_llm(
            client, label, type='ORGAN', location=None, model=model)

        # Step 2: Generate counterfactual → contradictory phrase
        contradictory_phrase = generate_counterfactual_phrase(
            client, label, type='ORGAN', location=None, candidate_class_root=candidate_class_root, model=model)

        # Step 3: Wrap into prompts
        supporting_prompt = _generate_prompt(supporting_phrase)
        contradictory_prompt = _generate_prompt(contradictory_phrase)

    else:  # Disease case: use organ aware prompting
        # Step 1: Humanize the label → supporting phrase
        location = label.split(
            'in the ')[-1].strip() if 'in the ' in label else 'the relevant organ'
        supporting_phrase = humanize_label_via_llm(
            client, label, type='LESION', location=location, model=model)

        # Step 2: Generate counterfactual → contradictory phrase
        contradictory_phrase = generate_counterfactual_phrase(
            client, supporting_phrase, type='LESION', location=location, candidate_class_root=candidate_class_root, model=model)

        # # Step 3: Get lesion location from knowledge base
        # _location = LESION_LOCATION.get(
        #     label.lower().strip(), location)

        # Step 3: Wrap into prompts
        supporting_prompt = _generate_prompt(supporting_phrase)
        contradictory_prompt = _generate_prompt(contradictory_phrase)

    # Step 4: Assemble result dict
    result = {
        'label': label,
        'supporting_phrase': supporting_phrase,
        'contradictory_phrase': contradictory_phrase,
        'supporting_prompt': supporting_prompt,
        'contradictory_prompt': contradictory_prompt,
    }

    return result

# ----------------------------
# Example usage
# ----------------------------


if __name__ == "__main__":
    test_labels = [
        "a normal left kidney",
        "a normal brain",
        "a tumor in the liver"
    ]

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_LLM_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "YOUR_LLM_BASE_URL")
    )
    model = "gpt-4.1-mini"
    candidate_class_root = os.getenv(
        "ENTITY_CANDIDATE_FILE", "YOUR_ENTITY_CANDIDATE_FILE"
    )
    for label in test_labels:
        cf = generate_supporting_and_counter_phrase(
            client, label, type=None, candidate_class_root=candidate_class_root, model=model)
        print(f"'{label}' → '{cf}'")
