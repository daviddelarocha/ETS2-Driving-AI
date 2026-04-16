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

CAPTURE_FPS = 10.0
JPEG_QUALITY = 75

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

CAPTURE_REGION = None
CAPTURE_MONITOR = 2

START_STOP_KEY = "!"
QUIT_KEY = ":"

FONT_SCALE = 0.55
FONT_THICKNESS = 1
LINE_STEP = 22

# Reference point for dataset size estimation
REFERENCE_ROWS = 8590
REFERENCE_SIZE_MB = 262.0
ESTIMATED_BYTES_PER_ROW = (REFERENCE_SIZE_MB * 1024 * 1024) / REFERENCE_ROWS


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_dataset_size_bytes(num_rows: int) -> int:
    return int(num_rows * ESTIMATED_BYTES_PER_ROW)


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
            "truck_game_steer",
            "truck_acceleration_x",
            "truck_acceleration_y",
            "truck_acceleration_z",
            "truck_engine_rpm",
            "truck_displayed_gear",
            "trailer_attached",
            "trailer_mass_kg",
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

    def get_num_samples(self) -> int:
        return max(0, self.index - 1)

    def write_sample(
        self,
        pil_image: Image.Image,
        timestamp: float,
        telemetry: Dict[str, float],
        controller: Dict[str, float],
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
            "steering": controller["steering"],
            "throttle": controller["throttle"],
            "brake": controller["brake"],
            "truck_speed_kmh": telemetry["truck_speed_kmh"],
            "speed_limit_kmh": telemetry["speed_limit_kmh"],
            "truck_game_steer": telemetry["truck_game_steer"],
            "truck_acceleration_x": telemetry["truck_acceleration_x"],
            "truck_acceleration_y": telemetry["truck_acceleration_y"],
            "truck_acceleration_z": telemetry["truck_acceleration_z"],
            "truck_engine_rpm": telemetry["truck_engine_rpm"],
            "truck_displayed_gear": telemetry["truck_displayed_gear"],
            "trailer_attached": telemetry["trailer_attached"],
            "trailer_mass_kg": telemetry["trailer_mass_kg"],
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
    controller: Dict[str, float],
    recording: bool,
    is_test: bool,
    recording_elapsed_seconds: float,
    dataset_size_bytes: int,
) -> np.ndarray:
    status_line = "TEST MODE" if is_test else ("RECORDING: ON" if recording else "RECORDING: OFF")

    lines = [
        status_line,
        (
            f"target pad | steer={controller['steering']:+.3f}  "
            f"thr={controller['throttle']:.3f}  "
            f"brk={controller['brake']:.3f}"
        ),
        (
            f"telemetry | speed={telemetry['truck_speed_kmh']:.1f} km/h  "
            f"limit={telemetry['speed_limit_kmh']:.1f} km/h  "
            f"gameSteer={telemetry['truck_game_steer']:+.3f}"
        ),
        (
            f"acc xyz | x={telemetry['truck_acceleration_x']:+.3f}  "
            f"y={telemetry['truck_acceleration_y']:+.3f}  "
            f"z={telemetry['truck_acceleration_z']:+.3f}"
        ),
        (
            f"engine | rpm={telemetry['truck_engine_rpm']:.0f}  "
            f"gear={telemetry['truck_displayed_gear']:.0f}"
        ),
        (
            f"trailer | attached={int(telemetry['trailer_attached'])}  "
            f"mass={telemetry['trailer_mass_kg']:.0f} kg"
        ),
        (
            f"rec={format_duration(recording_elapsed_seconds)}  "
            f"dataset~={format_bytes(dataset_size_bytes)}  "
            f"fps={CAPTURE_FPS:.1f}"
        ),
        "! = start/stop | : = exit",
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
        "targets_source": "switch_pro_controller",
        "dataset_size_estimation": {
            "reference_rows": REFERENCE_ROWS,
            "reference_size_mb": REFERENCE_SIZE_MB,
            "estimated_bytes_per_row": ESTIMATED_BYTES_PER_ROW,
        },
        "numeric_features": [
            "truck_speed_kmh",
            "speed_limit_kmh",
            "truck_game_steer",
            "truck_acceleration_x",
            "truck_acceleration_y",
            "truck_acceleration_z",
            "truck_engine_rpm",
            "truck_displayed_gear",
            "trailer_attached",
            "trailer_mass_kg",
        ],
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_test_mode(
    telemetry_adapter: HttpTelemetryAdapter,
    controller_adapter: SwitchProControllerAdapter,
) -> None:
    print("Modo test activo.")
    print("Pulsa ':' para salir.")

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

                telemetry = telemetry_adapter.read().to_dict()
                controller = controller_adapter.read().to_dict()

                # In test mode we don't scan the folder, just show 0 as estimate baseline
                dataset_size_bytes = 0

                frame_bgr = pil_to_bgr(processed_image)
                frame_bgr = draw_overlay(
                    frame_bgr=frame_bgr,
                    telemetry=telemetry,
                    controller=controller,
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
                        f"pad steer={controller['steering']:+.3f} "
                        f"thr={controller['throttle']:.3f} "
                        f"brk={controller['brake']:.3f} | "
                        f"speed={telemetry['truck_speed_kmh']:.1f} "
                        f"limit={telemetry['speed_limit_kmh']:.1f} "
                        f"gameSteer={telemetry['truck_game_steer']:+.3f} "
                        f"rpm={telemetry['truck_engine_rpm']:.0f} "
                        f"gear={telemetry['truck_displayed_gear']:.0f} "
                        f"trailer_attached={int(telemetry['trailer_attached'])} "
                        f"trailer_mass={telemetry['trailer_mass_kg']:.0f}"
                    )
                    last_print = now
    finally:
        cv2.destroyAllWindows()


def run_dataset_mode(
    telemetry_adapter: HttpTelemetryAdapter,
    controller_adapter: SwitchProControllerAdapter,
) -> None:
    writer = DatasetWriter(DATASET_DIR, IMAGES_DIR, CSV_PATH)
    writer.setup()
    save_meta()

    print(f"Dataset dir: {DATASET_DIR.resolve()}")
    print("Pulsa ! para empezar/parar grabación.")
    print("Pulsa : para salir.")

    recording = False
    toggle_pressed = False
    recording_started_at: float | None = None
    accumulated_recording_seconds = 0.0

    frame_interval = 1.0 / CAPTURE_FPS
    next_frame_time = time.perf_counter()

    try:
        with mss() as sct:
            while True:
                if keyboard.is_pressed(QUIT_KEY):
                    print("Saliendo...")
                    break

                if keyboard.is_pressed(START_STOP_KEY) or controller_adapter.read().to_dict()["toggle_autopilot"] > 0.5:
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

                telemetry = telemetry_adapter.read().to_dict()
                controller = controller_adapter.read().to_dict()

                now_time = time.time()
                if recording and recording_started_at is not None:
                    recording_elapsed_seconds = accumulated_recording_seconds + (now_time - recording_started_at)
                else:
                    recording_elapsed_seconds = accumulated_recording_seconds

                dataset_size_bytes = estimate_dataset_size_bytes(writer.get_num_samples())

                frame_bgr = pil_to_bgr(processed_image)
                frame_bgr = draw_overlay(
                    frame_bgr=frame_bgr,
                    telemetry=telemetry,
                    controller=controller,
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
                    controller=controller,
                )
    finally:
        writer.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura de dataset ETS2 con target desde mando")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Solo visualiza captura, mando y telemetría, sin guardar dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    telemetry_adapter = HttpTelemetryAdapter()
    controller_adapter = SwitchProControllerAdapter()

    telemetry_adapter.connect()
    controller_adapter.connect()

    try:
        if args.test:
            run_test_mode(telemetry_adapter, controller_adapter)
        else:
            run_dataset_mode(telemetry_adapter, controller_adapter)
    finally:
        controller_adapter.close()
        telemetry_adapter.close()


if __name__ == "__main__":
    main()