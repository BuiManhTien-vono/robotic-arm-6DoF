import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = "qwen-local"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_QWEN_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


class TargetLocalization(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target_object: str = Field(
        default="target_object",
        min_length=1,
        validation_alias=AliasChoices("target_object", "label", "object", "name"),
    )
    bbox: list[int] = Field(validation_alias=AliasChoices("bbox", "bbox_2d"))
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("bbox", mode="before")
    @classmethod
    def normalize_bbox(cls, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must be [x_min, y_min, x_max, y_max]")
        return [int(round(float(v))) for v in value]


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def _parse_json_response(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"VLM response is not valid JSON: {text}")


def _coerce_localization_payload(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("VLM response JSON list is empty.")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"VLM response must be a JSON object, got {type(parsed).__name__}.")
    return parsed


def _validate_result(parsed: Any, width: int, height: int) -> dict[str, Any]:
    payload = _coerce_localization_payload(parsed)
    result = TargetLocalization.model_validate(payload).model_dump()
    result["bbox"] = _clamp_bbox(result["bbox"], width, height)
    return result


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(0, min(width, x_max))
    y_max = max(0, min(height, y_max))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bbox after clamping: {[x_min, y_min, x_max, y_max]}")
    return [x_min, y_min, x_max, y_max]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _torch_dtype_from_env(torch_module: Any) -> Any:
    value = os.getenv("QWEN_VL_TORCH_DTYPE", "auto").strip().lower()
    if value in {"", "auto"}:
        return "auto"
    aliases = {
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "half": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported QWEN_VL_TORCH_DTYPE={value!r}. Use auto, float16, bfloat16, or float32.")
    return aliases[value]


def _max_memory_from_env() -> dict[Any, str] | None:
    gpu_memory = os.getenv("QWEN_VL_MAX_MEMORY_GPU")
    cpu_memory = os.getenv("QWEN_VL_MAX_MEMORY_CPU")
    if not gpu_memory and not cpu_memory:
        return None
    max_memory: dict[Any, str] = {}
    if gpu_memory:
        max_memory[0] = gpu_memory
    if cpu_memory:
        max_memory["cpu"] = cpu_memory
    return max_memory


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _normalize_backend(backend: str | None) -> str:
    value = (backend or os.getenv("VLM_BACKEND") or os.getenv("VLM_PROVIDER") or DEFAULT_BACKEND).strip().lower()
    aliases = {
        "local": "qwen-local",
        "qwen": "qwen-local",
        "qwen2.5-vl": "qwen-local",
        "qwen25vl": "qwen-local",
        "qwen-local": "qwen-local",
        "gemini": "gemini",
        "google": "gemini",
        "google-gemini": "gemini",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported VLM backend: {backend}") from exc


def _resolve_model(backend: str, model: str | None) -> str:
    if model:
        return model
    if backend == "qwen-local":
        return (
            os.getenv("QWEN_VL_MODEL")
            or os.getenv("LOCAL_VLM_MODEL")
            or os.getenv("VLM_MODEL")
            or DEFAULT_QWEN_MODEL
        )
    return os.getenv("GEMINI_MODEL") or os.getenv("VLM_MODEL") or DEFAULT_GEMINI_MODEL


def build_localization_prompt(
    user_command: str,
    image_width: int,
    image_height: int,
    extra_rules: str | None = None,
) -> str:
    extra_rules_text = f"\nExtra rules:\n{extra_rules.strip()}\n" if extra_rules else ""
    return f"""
You localize one target object for a robot grasping system.
The command may be Vietnamese or English.

Command: {user_command}

Return compact JSON only:
{{"target_object":"name","bbox":[x_min,y_min,x_max,y_max],"confidence":0.0,"reason":"short"}}

Rules:
- bbox is pixel coordinates on the original image, not normalized coordinates.
- image width={image_width}, height={image_height}.
- x range: 0..{image_width}; y range: 0..{image_height}.
- No markdown, no explanation outside JSON.
- Keep reason under 5 words.
{extra_rules_text}
""".strip()


def _missing_qwen_dependency_error(exc: ImportError) -> RuntimeError:
    message = (
        "Missing or incompatible local Qwen2.5-VL dependencies.\n"
        f"Original import error: {exc}\n\n"
        "Use a separate VLM environment. Install a PyTorch + torchvision build that match each other, then run:\n"
        "  python -m pip install transformers accelerate qwen-vl-utils\n"
        "Do not install this stack into the GraspNet environment if it uses an older torch build.\n"
        "For low VRAM, also install bitsandbytes and set QWEN_VL_4BIT=1."
    )
    return RuntimeError(message)


def _load_qwen_model_processor(
    *,
    model_id: str,
    min_pixels: int | None,
    max_pixels: int | None,
    device_map: str,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    load_in_8bit = _env_bool("QWEN_VL_8BIT", False)
    if load_in_4bit and load_in_8bit:
        raise ValueError("Use only one quantization mode: QWEN_VL_4BIT=1 or QWEN_VL_8BIT=1, not both.")

    cache_key = (model_id, min_pixels, max_pixels, device_map, load_in_4bit, load_in_8bit)
    if cache_key in _QWEN_CACHE:
        return _QWEN_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise _missing_qwen_dependency_error(exc) from exc

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype_from_env(torch),
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    max_memory = _max_memory_from_env()
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    attn_implementation = os.getenv("QWEN_VL_ATTENTION_IMPL")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QWEN_VL_4BIT=1 or QWEN_VL_8BIT=1 requires bitsandbytes and BitsAndBytesConfig.") from exc
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

    normalized_model_id = model_id.lower().replace("_", "-")
    try:
        if "qwen2-vl" in normalized_model_id and "qwen2.5" not in normalized_model_id:
            from transformers import Qwen2VLForConditionalGeneration

            model_cls = Qwen2VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise _missing_qwen_dependency_error(exc) from exc

    model = model_cls.from_pretrained(model_id, **model_kwargs)
    model.eval()

    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    _QWEN_CACHE[cache_key] = (model, processor)
    return model, processor


def _model_input_device(model: Any, torch_module: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _run_qwen_local(
    image_path: Path,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    device_map: str | None = None,
    load_in_4bit: bool | None = None,
) -> str:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise _missing_qwen_dependency_error(exc) from exc

    min_pixels = min_pixels if min_pixels is not None else _env_int("QWEN_VL_MIN_PIXELS")
    max_pixels = max_pixels if max_pixels is not None else _env_int("QWEN_VL_MAX_PIXELS")
    device_map = device_map or os.getenv("QWEN_VL_DEVICE_MAP")
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
    load_in_4bit = _env_bool("QWEN_VL_4BIT", False) if load_in_4bit is None else load_in_4bit
    max_new_tokens = max_new_tokens or _env_int("QWEN_VL_MAX_NEW_TOKENS") or 256

    model, processor = _load_qwen_model_processor(
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model_input_device(model, torch))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _run_gemini(
    image_path: Path,
    prompt: str,
    *,
    model_id: str,
    api_key: str | None = None,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini backend requires google-genai. Install it with: python -m pip install google-genai") from exc

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it in .env or use the default local backend with VLM_BACKEND=qwen-local."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type=_mime_type(image_path)),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return response.text or ""


def localize_target_object(
    image_path: str | os.PathLike[str],
    user_command: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    backend: str | None = None,
    extra_rules: str | None = None,
    max_new_tokens: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    device_map: str | None = None,
    load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    _load_env()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    width, height = _image_size(image_path)
    prompt = build_localization_prompt(user_command, width, height, extra_rules=extra_rules)
    selected_backend = _normalize_backend(backend)
    model_id = _resolve_model(selected_backend, model)

    if selected_backend == "qwen-local":
        raw_text = _run_qwen_local(
            image_path,
            prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
        )
    else:
        raw_text = _run_gemini(
            image_path,
            prompt,
            model_id=model_id,
            api_key=api_key,
        )

    parsed = _parse_json_response(raw_text)
    result = _validate_result(parsed, width, height)
    result["vlm_backend"] = selected_backend
    result["vlm_model"] = model_id
    return result
