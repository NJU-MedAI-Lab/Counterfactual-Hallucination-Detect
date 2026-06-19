import transformers
from dataclasses import dataclass, field
import os
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2

    sys_enable: bool = field(default=False)


@dataclass
class TestArguments:
    test_only: bool = field(default=False)
    do_entity_extract: bool = field(default=True)
    do_counterfactual_test: bool = field(default=True)
    entity_candidate_file: str = field(
        default_factory=lambda: os.getenv("ENTITY_CANDIDATE_FILE", "YOUR_ENTITY_CANDIDATE_FILE"))
    entity_extract_model: str = field(default="gpt-4.1-mini")
    cf_model: str = field(default="gpt-4.1-mini")
    uncertainty_type: str = field(default="all")
    test_batch_size: int = field(default=4)
    test_result_file: str = field(default="results.json")
    image_root_dir: str = field(
        default_factory=lambda: os.getenv("IMAGE_ROOT_DIR", "YOUR_IMAGE_ROOT_DIR"))
    sample_num: int = field(default=5)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    # Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
