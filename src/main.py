from __future__ import annotations

import argparse

from src.common.runtime import configure_runtime

configure_runtime()

from src.app.orchestrator import Orchestrator
from src.camera.camera_service import CameraService
from src.common.config import load_app_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartReach-HSM local prototype")
    parser.add_argument(
        "--text-command",
        type=str,
        default=None,
        help="Direct text command for local testing",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Audio file path for ASR testing",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Static image path for local testing",
    )
    parser.add_argument("--camera-id", type=int, default=None, help="Camera device id")
    parser.add_argument(
        "--session-steps",
        type=int,
        default=None,
        help="Number of screenshot loop iterations",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app_cfg = load_app_config()
    camera_id = (
        args.camera_id
        if args.camera_id is not None
        else app_cfg["capture"]["camera_id"]
    )
    session_steps = (
        args.session_steps
        if args.session_steps is not None
        else app_cfg["capture"]["session_steps"]
    )
    capture_cfg = app_cfg.get("capture", {})
    camera_service = CameraService(
        camera_id=camera_id,
        mirror_correction=bool(capture_cfg.get("mirror_correction", False)),
        stream_fps=float(capture_cfg.get("stream_fps", 20.0)),
    )
    orchestrator = Orchestrator(camera_service=camera_service)
    try:
        orchestrator.run(
            text_command=args.text_command,
            audio_path=args.audio_path,
            image_path=args.image_path,
            session_steps=session_steps,
        )
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
