import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_localizer import TargetLocalization, localize_target_object


DEFAULT_IMAGE = PROJECT_ROOT / "graspnet-baseline" / "doc" / "example_data" / "color.png"
DEFAULT_OUTPUT_IMAGE = PROJECT_ROOT / "data" / "outputs" / "vlm_bbox.png"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "outputs" / "vlm_result.json"


def draw_bbox(image_path: Path, result: dict, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    x_min, y_min, x_max, y_max = result["bbox"]
    label = f'{result["target_object"]} {result["confidence"]:.2f}'
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(
        image,
        label,
        (x_min, max(24, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test VLM target localization.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="RGB image path")
    parser.add_argument(
        "--command",
        default="Hay gap vat the phu hop nhat trong canh.",
        help="User grasping command",
    )
    parser.add_argument("--output-image", default=str(DEFAULT_OUTPUT_IMAGE))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--vlm-backend", default=None, help="VLM backend: qwen-local or gemini. Default: VLM_BACKEND or qwen-local.")
    parser.add_argument("--vlm-model", default=None, help="Model id/name for the selected VLM backend.")
    parser.add_argument("--extra-rules", default=None, help="Additional localization rules appended to the VLM prompt.")
    parser.add_argument(
        "--mock-bbox",
        nargs=4,
        type=int,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Skip VLM and test bbox drawing with a manual bbox.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    if args.mock_bbox:
        result = TargetLocalization(
            target_object="manual_target",
            bbox=args.mock_bbox,
            confidence=1.0,
            reason="manual bbox for local drawing test",
        ).model_dump()
    else:
        result = localize_target_object(
            image_path,
            args.command,
            backend=args.vlm_backend,
            model=args.vlm_model,
            extra_rules=args.extra_rules,
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_bbox(image_path, result, Path(args.output_image))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {output_json}")
    print(f"Saved bbox image: {args.output_image}")


if __name__ == "__main__":
    main()
