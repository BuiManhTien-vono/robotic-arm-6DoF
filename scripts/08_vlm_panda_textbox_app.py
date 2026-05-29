import argparse
import ctypes
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import BooleanVar, Button, Checkbutton, END, Label, Text, Tk, messagebox
from tkinter.scrolledtext import ScrolledText


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_localizer import localize_target_object


PANDA_SIM_SCRIPT = PROJECT_ROOT / "scripts" / "06_run_panda_pick_place_sim.py"
VLM_PANDA_SCRIPT = PROJECT_ROOT / "scripts" / "07_run_vlm_panda_pick_place.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "vlm_panda_textbox"
DEFAULT_VLM_PYTHON = PROJECT_ROOT / ".venv_vlm" / "Scripts" / "python.exe"
VLM_TEST_SCRIPT = PROJECT_ROOT / "scripts" / "01_test_vlm.py"
WINDOWS_ACCESS_VIOLATION = 3221225477


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_memory_gb() -> float | None:
    if os.name != "nt":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / (1024**3)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panda_module = load_module(PANDA_SIM_SCRIPT, "panda_pick_place_sim")
vlm_panda = load_module(VLM_PANDA_SCRIPT, "vlm_panda_pick_place")


class VlmPandaTextboxApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.graspnet_adapter = None

        self.sim = panda_module.PandaPickPlaceSim(
            gui=True,
            realtime_sleep=True,
            seed=args.seed,
            grasp_assist=True,
            motion_slowdown=args.speed_scale,
        )
        self.sim.connect()
        self.sim.setup_scene(args.num_objects)

        self.root = Tk()
        self.root.title("VLM Panda Pick-and-Place")
        self.root.geometry("720x460")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        Label(self.root, text="Nhap lenh cho robot:").pack(anchor="w", padx=12, pady=(10, 4))
        self.command_box = Text(self.root, height=3, wrap="word")
        self.command_box.pack(fill="x", padx=12)
        self.command_box.insert("1.0", args.default_command)

        self.mock_var = BooleanVar(value=args.mock)
        Checkbutton(
            self.root,
            text="Test nhanh khong goi VLM: chon object dau tien trong scene",
            variable=self.mock_var,
        ).pack(anchor="w", padx=12, pady=(8, 0))

        self.fast_semantic_var = BooleanVar(value=args.fast_semantic)
        Checkbutton(
            self.root,
            text="Fast sim semantic: khop mau/hinh dang truoc khi goi Qwen",
            variable=self.fast_semantic_var,
        ).pack(anchor="w", padx=12, pady=(4, 0))

        self.use_graspnet_var = BooleanVar(value=not args.no_graspnet)
        Checkbutton(
            self.root,
            text="Use GraspNet trained checkpoint for grasp pose",
            variable=self.use_graspnet_var,
        ).pack(anchor="w", padx=12, pady=(4, 0))

        self.run_button = Button(self.root, text="Run Command", command=self.run_command)
        self.run_button.pack(anchor="w", padx=12, pady=(8, 0))

        self.reset_button = Button(self.root, text="Reset Scene", command=self.reset_scene)
        self.reset_button.pack(anchor="w", padx=12, pady=(6, 0))

        Label(self.root, text="Log:").pack(anchor="w", padx=12, pady=(12, 4))
        self.log_box = ScrolledText(self.root, height=12, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.root.after(100, self.poll_log_queue)
        self.log("Ready. Nhap lenh vao textbox roi bam Run Command.")

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def poll_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_box.configure(state="normal")
            self.log_box.insert(END, message + "\n")
            self.log_box.see(END)
            self.log_box.configure(state="disabled")
        self.root.after(100, self.poll_log_queue)

    def run(self) -> None:
        self.root.mainloop()

    def run_command(self) -> None:
        if self.running:
            return
        command = self.command_box.get("1.0", END).strip()
        if not command:
            messagebox.showwarning("Missing command", "Hay nhap lenh cho robot.")
            return
        use_mock = bool(self.mock_var.get())
        use_fast_semantic = bool(self.fast_semantic_var.get())
        use_graspnet = bool(self.use_graspnet_var.get())
        self.running = True
        self.run_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        thread = threading.Thread(
            target=self.execute_command,
            args=(command, use_mock, use_fast_semantic, use_graspnet),
            daemon=True,
        )
        thread.start()

    def execute_command(
        self,
        command: str,
        use_mock: bool,
        use_fast_semantic: bool,
        use_graspnet: bool,
    ) -> None:
        try:
            self.log(f"Command: {command}")
            render_data = vlm_panda.render_camera_data(
                width=self.args.camera_width,
                height=self.args.camera_height,
            )
            rgb = render_data["rgb"]
            segmentation = render_data["segmentation"]
            render_path = self.output_dir / "01_render_rgb.png"
            vlm_panda.save_rgb(rgb, render_path)
            vlm_panda.save_depth_visual(render_data["depth_m"], self.output_dir / "01_render_depth.png")
            self.log(f"Rendered camera image: {render_path}")

            if use_mock:
                object_id = self.sim.object_ids[0]
                bbox = vlm_panda.bbox_from_object(segmentation, object_id)
                vlm_result = {
                    "target_object": f"mock_object_{object_id}",
                    "bbox": bbox,
                    "confidence": 1.0,
                    "reason": "mock bbox from PyBullet segmentation",
                }
                self.log("Mock mode: using first object bbox instead of VLM.")
            else:
                vlm_result = None
                if use_fast_semantic:
                    vlm_result = self.fast_semantic_localization(segmentation, command)
                    if vlm_result is not None:
                        self.log(
                            "Fast semantic matched PyBullet object: "
                            f"{vlm_result['target_object']} bbox={vlm_result['bbox']}"
                        )
                    else:
                        self.log("Fast semantic did not find a confident match; falling back to Qwen.")
                if vlm_result is None:
                    self.log(f"Calling VLM backend: {self.args.vlm_backend or 'qwen-local'}...")
                    extra_rules = (
                        "- Select only one small movable object resting on the gray table.\n"
                        "- Do not select the robot arm, gripper, floor, table, gray platform, or blue bin/tray.\n"
                        "- If the user asks for a color and the blue tray/bin matches, ignore the tray/bin and choose the best small tabletop object instead."
                    )
                    vlm_result = self.localize_with_configured_backend(
                        render_path,
                        command,
                        extra_rules=extra_rules,
                    )

            (self.output_dir / "02_vlm_result.json").write_text(
                json.dumps(vlm_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.log(f"VLM target: {vlm_result['target_object']} bbox={vlm_result['bbox']} confidence={vlm_result['confidence']}")

            selection = vlm_panda.select_object_from_bbox(
                segmentation,
                vlm_result["bbox"],
                self.sim.object_ids,
                object_metadata=self.sim.object_metadata,
                text_hint=f'{command} {vlm_result.get("target_object", "")} {vlm_result.get("reason", "")}',
            )
            selected_object_id = selection["object_id"]
            vlm_panda.draw_bbox(
                rgb,
                selection.get("selected_object_bbox", selection["bbox_used"]),
                f'{vlm_result["target_object"]} -> id {selected_object_id}',
                self.output_dir / "03_selected_bbox.png",
            )
            self.log(f"Selected PyBullet object_id: {selected_object_id}")
            if selection.get("selected_metadata"):
                self.log(f"Selected object: {selection['selected_metadata']}")
            if selection.get("fallback_used"):
                self.log(f"Selection fallback: {selection.get('fallback_reason')}")

            if use_graspnet:
                self.log("Running GraspNet trained checkpoint...")
                graspnet_summary, self.graspnet_adapter = vlm_panda.run_graspnet_for_object(
                    render_data,
                    selected_object_id,
                    output_dir=self.output_dir / "04_graspnet",
                    checkpoint_path=self.args.checkpoint,
                    adapter=self.graspnet_adapter,
                    num_point=self.args.num_point,
                    num_view=self.args.num_view,
                    top_k=self.args.top_k,
                    collision_thresh=self.args.collision_thresh,
                    voxel_size=self.args.voxel_size,
                )
                self.log(
                    "GraspNet best grasp: "
                    f"score={graspnet_summary['best_grasp_camera']['score']:.4f}, "
                    f"world={graspnet_summary['best_grasp_world']['translation']}"
                )
                self.log("Robot executing GraspNet-based pick-and-place...")
                pick_result = self.sim.pick_and_place_with_graspnet_pose(
                    selected_object_id,
                    graspnet_summary["best_grasp_world"],
                    use_grasp_orientation=self.args.use_graspnet_orientation,
                )
            else:
                graspnet_summary = {"status": "skipped", "reason": "UI GraspNet checkbox disabled"}
                self.log("Robot executing heuristic pick-and-place...")
                pick_result = self.sim.pick_and_place(selected_object_id)
            summary = {
                "status": "completed",
                "command": command,
                "vlm_result": vlm_result,
                "selection": selection,
                "graspnet": graspnet_summary,
                "pick_place": pick_result,
            }
            summary_path = self.output_dir / "vlm_panda_textbox_result.json"
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log(f"Pick result: {pick_result}")
            self.log(f"Saved result: {summary_path}")
        except Exception as exc:
            self.log(f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_button.configure(state="normal"))
            self.root.after(0, lambda: self.reset_button.configure(state="normal"))

    def fast_semantic_localization(self, segmentation, command: str) -> dict | None:
        visible_bboxes = vlm_panda.visible_object_bboxes(segmentation, self.sim.object_ids)
        normalized_command = vlm_panda.normalize_text(command)
        specific_xanh_terms = ("xanh la", "xanh luc", "xanh duong", "xanh lam", "xanh ngoc")
        generic_xanh = (
            vlm_panda.has_text_term(normalized_command, "xanh")
            and not any(vlm_panda.has_text_term(normalized_command, term) for term in specific_xanh_terms)
        )

        scored = []
        for object_id, bbox in visible_bboxes.items():
            metadata = self.sim.object_metadata.get(int(object_id), {})
            score = vlm_panda.semantic_score(metadata, command)
            color_name = str(metadata.get("color_name", ""))
            if generic_xanh:
                if color_name == "blue":
                    score += 6
                elif color_name in {"green", "cyan"}:
                    score += 3
            scored.append((int(score), int(metadata.get("index", object_id)), int(object_id), bbox, metadata))

        if not scored:
            return None

        best_score, _, object_id, bbox, metadata = max(scored, key=lambda item: (item[0], -item[1]))
        if best_score <= 0:
            return None

        color_name = str(metadata.get("color_name", "")).strip()
        shape = str(metadata.get("shape", metadata.get("type", ""))).strip()
        target_object = " ".join(part for part in (color_name, shape) if part) or f"object_{object_id}"
        return {
            "target_object": target_object,
            "bbox": bbox,
            "confidence": min(0.98, 0.55 + best_score / 40.0),
            "reason": f"fast PyBullet semantic match, score={best_score}",
            "vlm_backend": "fast-sim-semantic",
            "vlm_model": "pybullet-metadata",
            "object_id_hint": object_id,
        }

    def localize_with_configured_backend(
        self,
        image_path: Path,
        command: str,
        *,
        extra_rules: str,
    ) -> dict:
        backend = (self.args.vlm_backend or "qwen-local").lower()
        if backend in {"qwen-local", "qwen", "local"} and self.args.vlm_subprocess:
            return self.localize_qwen_subprocess(image_path, command, extra_rules=extra_rules)
        return localize_target_object(
            image_path,
            command,
            backend=self.args.vlm_backend,
            model=self.args.vlm_model,
            extra_rules=extra_rules,
        )

    def localize_qwen_subprocess(
        self,
        image_path: Path,
        command: str,
        *,
        extra_rules: str,
    ) -> dict:
        vlm_python = Path(self.args.vlm_python)
        if not vlm_python.exists():
            raise FileNotFoundError(f"Missing VLM Python environment: {vlm_python}")

        output_json = self.output_dir / "02_vlm_result.json"
        output_image = self.output_dir / "02_vlm_bbox_raw.png"
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env.setdefault("QWEN_VL_MAX_PIXELS", str(self.args.vlm_max_pixels))
        env.setdefault("QWEN_VL_MAX_NEW_TOKENS", str(self.args.vlm_max_new_tokens))
        env.setdefault("QWEN_VL_DEVICE_MAP", self.args.vlm_device_map)
        if self.args.vlm_offline:
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")

        cmd = [
            str(vlm_python),
            str(VLM_TEST_SCRIPT),
            "--image",
            str(image_path),
            "--command",
            command,
            "--output-json",
            str(output_json),
            "--output-image",
            str(output_image),
            "--vlm-backend",
            "qwen-local",
            "--extra-rules",
            extra_rules,
        ]
        if self.args.vlm_model:
            cmd.extend(["--vlm-model", self.args.vlm_model])

        self.log(f"Qwen subprocess: {vlm_python}")
        self.log(
            "Qwen settings: "
            f"model={self.args.vlm_model or 'default'}, "
            f"device_map={env['QWEN_VL_DEVICE_MAP']}, "
            f"max_pixels={env['QWEN_VL_MAX_PIXELS']}, "
            f"max_new_tokens={env['QWEN_VL_MAX_NEW_TOKENS']}, "
            f"offline={self.args.vlm_offline}"
        )
        free_ram_gb = available_memory_gb()
        if free_ram_gb is not None:
            self.log(f"System RAM available before Qwen: {free_ram_gb:.2f} GB.")
        self.log("Qwen is running. CPU mode can take several minutes.")

        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        lines: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line.rstrip())

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        start_time = time.monotonic()
        next_heartbeat = start_time + 30.0
        while process.poll() is None:
            while True:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    break
                if line:
                    self.log(f"[Qwen] {line}")
            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = int(now - start_time)
                self.log(f"[Qwen] still running... elapsed {elapsed}s")
                next_heartbeat = now + 30.0
            time.sleep(0.2)

        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            if line:
                self.log(f"[Qwen] {line}")

        elapsed = int(time.monotonic() - start_time)
        if process.returncode != 0:
            if process.returncode == WINDOWS_ACCESS_VIOLATION:
                free_ram_note = ""
                free_ram_gb = available_memory_gb()
                if free_ram_gb is not None:
                    free_ram_note = f" Current available RAM: {free_ram_gb:.2f} GB."
                raise RuntimeError(
                    "Qwen crashed with Windows access violation 0xC0000005 while loading the model. "
                    "On this machine this usually means the CPU model load ran out of usable RAM."
                    f"{free_ram_note} Close other apps or use GPU/quantized/smaller VLM."
                )
            raise RuntimeError(f"Qwen subprocess failed after {elapsed}s with exit code {process.returncode}.")
        self.log(f"Qwen finished in {elapsed}s.")

        if not output_json.exists():
            raise FileNotFoundError(f"Qwen did not create output JSON: {output_json}")
        return json.loads(output_json.read_text(encoding="utf-8"))

    def reset_scene(self) -> None:
        if self.running:
            return
        self.running = True
        self.run_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        thread = threading.Thread(target=self._reset_scene_worker, daemon=True)
        thread.start()

    def _reset_scene_worker(self) -> None:
        try:
            self.log("Resetting PyBullet scene...")
            self.sim.disconnect()
            self.sim = panda_module.PandaPickPlaceSim(
                gui=True,
                realtime_sleep=True,
                seed=self.args.seed,
                grasp_assist=True,
                motion_slowdown=self.args.speed_scale,
            )
            self.sim.connect()
            self.sim.setup_scene(self.args.num_objects)
            self.log("Scene reset done.")
        except Exception as exc:
            self.log(f"ERROR during reset: {type(exc).__name__}: {exc}")
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_button.configure(state="normal"))
            self.root.after(0, lambda: self.reset_button.configure(state="normal"))

    def close(self) -> None:
        try:
            self.sim.disconnect()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Textbox UI for VLM-controlled Panda pick-and-place.")
    parser.add_argument("--num-objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed-scale", type=float, default=2.5, help="Motion slowdown multiplier. Larger is slower.")
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--mock", action="store_true", help="Start with mock mode enabled.")
    parser.add_argument(
        "--fast-semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PyBullet object metadata for color/shape commands before calling Qwen.",
    )
    parser.add_argument("--vlm-backend", default=None, help="VLM backend: qwen-local or gemini. Default: VLM_BACKEND or qwen-local.")
    parser.add_argument("--vlm-model", default=None, help="Model id/name for the selected VLM backend.")
    parser.add_argument("--vlm-subprocess", action=argparse.BooleanOptionalAction, default=True, help="Run local Qwen in .venv_vlm subprocess.")
    parser.add_argument("--vlm-python", default=str(DEFAULT_VLM_PYTHON), help="Python executable for the local Qwen environment.")
    parser.add_argument("--vlm-device-map", default="cpu", help="Qwen device_map used in subprocess, e.g. cpu or auto.")
    parser.add_argument("--vlm-offline", action=argparse.BooleanOptionalAction, default=True, help="Use local HuggingFace cache without HEAD requests.")
    parser.add_argument("--vlm-max-pixels", type=int, default=50176)
    parser.add_argument("--vlm-max-new-tokens", type=int, default=80)
    parser.add_argument("--checkpoint", default=str(vlm_panda.DEFAULT_CHECKPOINT))
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--no-graspnet", action="store_true", help="Start with GraspNet disabled.")
    parser.add_argument("--use-graspnet-orientation", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--default-command", default="Hay gap mot vat the nho tren ban.")
    return parser.parse_args()


def main() -> None:
    app = VlmPandaTextboxApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
