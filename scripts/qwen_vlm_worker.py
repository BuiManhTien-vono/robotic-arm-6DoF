import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_localizer import localize_target_object, preload_qwen_local


def draw_bbox(image_path: Path, result: dict[str, Any], output_path: Path) -> None:
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


def write_response(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    start_time = time.monotonic()
    image_path = Path(request["image"])
    output_json = Path(request["output_json"])
    output_image = Path(request["output_image"])

    result = localize_target_object(
        image_path,
        request["command"],
        backend="qwen-local",
        model=request.get("model"),
        extra_rules=request.get("extra_rules"),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_bbox(image_path, result, output_image)
    return {
        "status": "ok",
        "elapsed_s": round(time.monotonic() - start_time, 3),
        "result": result,
        "output_json": str(output_json),
        "output_image": str(output_image),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent local Qwen VLM JSONL worker.")
    parser.add_argument("--ready-message", default="Qwen worker ready.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Qwen worker process started.", file=sys.stderr, flush=True)
    if args.preload:
        if not args.model:
            raise ValueError("--preload requires --model")
        start_time = time.monotonic()
        print(f"Preloading Qwen model: {args.model}", file=sys.stderr, flush=True)
        try:
            preload_qwen_local(model_id=args.model)
            print(f"Qwen model preloaded in {time.monotonic() - start_time:.1f}s.", file=sys.stderr, flush=True)
        except Exception:
            print("Qwen preload failed. The worker will stay alive and retry on the next request.", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
    print(args.ready_message, file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__quit__":
            break
        try:
            request = json.loads(line)
            write_response(handle_request(request))
        except Exception as exc:
            write_response(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )


if __name__ == "__main__":
    main()
