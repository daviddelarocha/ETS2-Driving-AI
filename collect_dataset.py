from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import cv2
import keyboard
import numpy as np
from mss import mss
from PIL import Image

from controller_adapter import SwitchProControllerAdapter
from telemetry_adapter import HttpTelemetryAdapter


DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
CSV_PATH = DATASET_DIR / "samples.csv"
META_PATH = DATASET_DIR / "meta.json"
TRUCK_CONFIG_PATH = Path("truck_config.json")

CAPTURE_FPS = 10.0
JPEG_QUALITY = 75

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

CAPTURE_REGION = None
CAPTURE_MONITOR = 1

START_STOP_KEY = "!"
QUIT_KEY = "esc"

FONT_SCALE = 0.55
FONT_THICKNESS = 1
LINE_STEP = 22


def load_truck_config(path: Path = TRUCK_CONFIG_PATH) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "truck_power_hp": float(data["truck_power_hp"]),
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


class DatasetWriter:
    def __init__(self, dataset_dir: Path, images_dir: Path, csv_path: Path) -> None:
        self.dataset_dir = dataset_dir
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.index = 0
        self.csv_file = None
        self.csv_writer = None

    def setup(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        file_exists = self.csv_path.exists()
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")

        fieldnames = [
            "sample_id",
            "timestamp",
            "image_path",
            "steering",
            "throttle",
            "brake",
            "truck_speed_kmh",
            "speed_limit_kmh",
            "cargo_mass_kg",
            "truck_power_hp",
        ]

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)

        if not file_exists:
            self.csv_writer.writeheader()

        self.index = self._infer_next_index()

    def _infer_next_index(self) -> int:
        existing = sorted(self.images_dir.glob("*.jpg"))
        if not existing:
            return 1
        last = existing[-1].stem
        try:
            return int(last) + 1
        except ValueError:
            return len(existing) + 1

    def write_sample(
        self,
        pil_image: Image.Image,
        timestamp: float,
        telemetry: Dict[str, float],
    ) -> None:
        if self.csv_writer is None:
            raise RuntimeError("DatasetWriter no inicializado")

        sample_id = f"{self.index:08d}"
        image_filename = f"{sample_id}.jpg"
        image_path = self.images_dir / image_filename

        pil_image.save(image_path, format="JPEG", quality=JPEG_QUALITY)

        row = {
            "sample_id": sample_id,
            "timestamp": timestamp,
            "image_path": f"images/{image_filename}",
            "steering": telemetry["game_steer"],
            "throttle": telemetry["game_throttle"],
            "brake": telemetry["game_brake"],
            "truck_speed_kmh": telemetry["truck_speed_kmh"],
            "speed_limit_kmh": telemetry["speed_limit_kmh"],
            "cargo_mass_kg": telemetry["cargo_mass_kg"],
            "truck_power_hp": telemetry["truck_power_hp"],
        }

        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.index += 1

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()


def grab_frame(sct: mss, region: Dict[str, int] | None = None) -> Image.Image:
    if region is None:
        raw = sct.grab(sct.monitors[CAPTURE_MONITOR])
    else:
        raw = sct.grab(region)
    return Image.frombytes("RGB", raw.size, raw.rgb)


def preprocess_frame(image: Image.Image) -> Image.Image:
    return image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_telemetry_frame(raw_telemetry: Dict[str, float], truck_cfg: Dict[str, float]) -> Dict[str, float]:
    return {
        "truck_speed_kmh": raw_telemetry["truck_speed_kmh"],
        "speed_limit_kmh": raw_telemetry["speed_limit_kmh"],
        "cargo_mass_kg": raw_telemetry["cargo_mass_kg"],
        "truck_power_hp": truck_cfg["truck_power_hp"],
        "game_steer": raw_telemetry["game_steer"],
        "game_throttle": raw_telemetry["game_throttle"],
        "game_brake": raw_telemetry["game_brake"],
    }


def draw_text_lines(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    y = 22

    for line in lines:
        cv2.putText(
            out,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 0),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y += LINE_STEP

    return out


def draw_overlay(
    frame_bgr: np.ndarray,
    telemetry: Dict[str, float],
    recording: bool,
    is_test: bool,
    recording_elapsed_seconds: float,
    dataset_size_bytes: int,
) -> np.ndarray:
    status_line = "TEST MODE" if is_test else ("RECORDING: ON" if recording else "RECORDING: OFF")

    lines = [
        status_line,
        (
            f"targets | steer={telemetry['game_steer']:+.3f}  "
            f"thr={telemetry['game_throttle']:.3f}  "
            f"brk={telemetry['game_brake']:.3f}"
        ),
        (
            f"telemetry | speed={telemetry['truck_speed_kmh']:.1f} km/h  "
            f"limit={telemetry['speed_limit_kmh']:.1f} km/h"
        ),
        (
            f"truck | power={telemetry['truck_power_hp']:.0f} hp  "
            f"cargo={telemetry['cargo_mass_kg']:.0f} kg"
        ),
        (
            f"rec={format_duration(recording_elapsed_seconds)}  "
            f"dataset={format_bytes(dataset_size_bytes)}  "
            f"fps={CAPTURE_FPS:.1f}"
        ),
        "R = start/stop | ESC = exit",
    ]

    return draw_text_lines(frame_bgr, lines)


def save_meta() -> None:
    meta = {
        "created_at": time.time(),
        "capture_region": CAPTURE_REGION,
        "capture_monitor": CAPTURE_MONITOR,
        "capture_fps": CAPTURE_FPS,
        "output_width": OUTPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "targets_source": "telemetry.gameSteer/gameThrottle/gameBrake",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_test_mode(
    controller: SwitchProControllerAdapter,
    telemetry_adapter: HttpTelemetryAdapter,
    truck_cfg: Dict[str, float],
) -> None:
    print("Modo test activo.")
    print("Pulsa ESC para salir.")

    frame_interval = 1.0 / CAPTURE_FPS
    next_frame_time = time.perf_counter()
    last_print = 0.0

    try:
        with mss() as sct:
            while True:
                if keyboard.is_pressed(QUIT_KEY):
                    print("Saliendo de modo test...")
                    break

                now_perf = time.perf_counter()
                if now_perf < next_frame_time:
                    time.sleep(0.001)
                    continue
                next_frame_time += frame_interval

                image = grab_frame(sct, CAPTURE_REGION)
                processed_image = preprocess_frame(image)

                _ = controller.read()
                raw_telemetry = telemetry_adapter.read().to_dict()
                telemetry = build_telemetry_frame(raw_telemetry, truck_cfg)

                dataset_size_bytes = get_directory_size_bytes(DATASET_DIR)

                frame_bgr = pil_to_bgr(processed_image)
                frame_bgr = draw_overlay(
                    frame_bgr=frame_bgr,
                    telemetry=telemetry,
                    recording=False,
                    is_test=True,
                    recording_elapsed_seconds=0.0,
                    dataset_size_bytes=dataset_size_bytes,
                )

                cv2.imshow("ETS2 Capture Test", frame_bgr)
                cv2.waitKey(1)

                now = time.time()
                if now - last_print >= 0.5:
                    print(
                        f"target steer={telemetry['game_steer']:+.3f} "
                        f"thr={telemetry['game_throttle']:.3f} "
                        f"brk={telemetry['game_brake']:.3f} | "
                        f"speed={telemetry['truck_speed_kmh']:.1f} "
                        f"limit={telemetry['speed_limit_kmh']:.1f} "
                        f"cargo={telemetry['cargo_mass_kg']:.0f} "
                        f"power={telemetry['truck_power_hp']:.0f}"
                    )
                    last_print = now
    finally:
        cv2.destroyAllWindows()


def run_dataset_mode(
    controller: SwitchProControllerAdapter,
    telemetry_adapter: HttpTelemetryAdapter,
    truck_cfg: Dict[str, float],
) -> None:
    writer = DatasetWriter(DATASET_DIR, IMAGES_DIR, CSV_PATH)
    writer.setup()
    save_meta()

    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    print("Pulsa R para empezar/parar grabación.")
    print("Pulsa ESC para salir.")

    recording = False
    toggle_pressed = False
    recording_started_at: float | None = None
    accumulated_recording_seconds = 0.0

    frame_interval = 1.0 / CAPTURE_FPS
    next_frame_time = time.perf_counter()

    last_size_check = 0.0
    dataset_size_bytes = get_directory_size_bytes(DATASET_DIR)

    try:
        with mss() as sct:
            while True:
                if keyboard.is_pressed(QUIT_KEY):
                    print("Saliendo...")
                    break

                if keyboard.is_pressed(START_STOP_KEY):
                    if not toggle_pressed:
                        recording = not recording
                        if recording:
                            recording_started_at = time.time()
                        else:
                            if recording_started_at is not None:
                                accumulated_recording_seconds += time.time() - recording_started_at
                                recording_started_at = None
                        print("Grabación:", "ON" if recording else "OFF")
                        toggle_pressed = True
                else:
                    toggle_pressed = False

                now_perf = time.perf_counter()
                if now_perf < next_frame_time:
                    time.sleep(0.001)
                    continue
                next_frame_time += frame_interval

                image = grab_frame(sct, CAPTURE_REGION)
                processed_image = preprocess_frame(image)

                _ = controller.read()
                raw_telemetry = telemetry_adapter.read().to_dict()
                telemetry = build_telemetry_frame(raw_telemetry, truck_cfg)

                now_time = time.time()
                if recording and recording_started_at is not None:
                    recording_elapsed_seconds = accumulated_recording_seconds + (now_time - recording_started_at)
                else:
                    recording_elapsed_seconds = accumulated_recording_seconds

                if now_time - last_size_check >= 1.0:
                    dataset_size_bytes = get_directory_size_bytes(DATASET_DIR)
                    last_size_check = now_time

                frame_bgr = pil_to_bgr(processed_image)
                frame_bgr = draw_overlay(
                    frame_bgr=frame_bgr,
                    telemetry=telemetry,
                    recording=recording,
                    is_test=False,
                    recording_elapsed_seconds=recording_elapsed_seconds,
                    dataset_size_bytes=dataset_size_bytes,
                )

                cv2.imshow("ETS2 Dataset Capture", frame_bgr)
                cv2.waitKey(1)

                if not recording:
                    continue

                timestamp = time.time()
                writer.write_sample(
                    pil_image=processed_image,
                    timestamp=timestamp,
                    telemetry=telemetry,
                )
    finally:
        writer.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura de dataset multimodal ETS2")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Solo visualiza captura y telemetría, sin guardar dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[Setup] Loading truck config...")
    truck_cfg = load_truck_config()

    print("[Setup] Connecting controller...")
    controller = SwitchProControllerAdapter()
    controller.connect()

    print("[Setup] Connecting telemetry adapter...")
    telemetry_adapter = HttpTelemetryAdapter()
    telemetry_adapter.connect()

    try:
        if args.test:
            run_test_mode(controller, telemetry_adapter, truck_cfg)
        else:
            run_dataset_mode(controller, telemetry_adapter, truck_cfg)
    finally:
        telemetry_adapter.close()
        controller.close()


if __name__ == "__main__":
    main()