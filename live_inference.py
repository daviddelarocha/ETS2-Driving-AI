from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import cv2
import keyboard
import numpy as np
import torch
from mss import mss
from PIL import Image
from torchvision import transforms

from telemetry_adapter import HttpTelemetryAdapter
from model import DrivingModel
from controller_adapter import SwitchController, VirtualXboxController


# =========================
# Config
# =========================

MODEL_PATH = Path("artifacts/best_model.pt")

CAPTURE_MONITOR = 2
CAPTURE_REGION = None

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

QUIT_KEY = ":"

# Telemetry normalization
MAX_SPEED = 130.0
MAX_RPM = 3000.0
MAX_GEAR = 12.0
MAX_TRAILER_MASS = 50000.0

# Optional output smoothing
EMA_ALPHA = 0.30

FONT_SCALE = 0.55
FONT_THICKNESS = 1
LINE_STEP = 22

MODEL_VIEW_WINDOW = "Autopilot Status"
DEBUG_VIEW_WINDOW = "ETS2 Live Inference"

STATUS_WIDTH = 260
STATUS_HEIGHT = 90
STATUS_FONT_SCALE = 0.70
STATUS_FONT_THICKNESS = 2


# =========================
# Helpers
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETS2 live inference with virtual Xbox controller")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the debug overlay window",
    )
    parser.add_argument(
        "--manual-passthrough",
        action="store_true",
        help="When autopilot is OFF, forward the real controller to the virtual controller",
    )
    return parser.parse_args()


def grab_frame(sct: mss, region: Dict[str, int] | None = None) -> Image.Image:
    if region is None:
        raw = sct.grab(sct.monitors[CAPTURE_MONITOR])
    else:
        raw = sct.grab(region)
    return Image.frombytes("RGB", raw.size, raw.rgb)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_for_display(image: Image.Image) -> Image.Image:
    return image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.Resampling.LANCZOS)


def build_numeric_features(raw: Dict[str, float]) -> torch.Tensor:
    features = torch.tensor(
        [
            float(raw["truck_speed_kmh"]) / MAX_SPEED,
            float(raw["speed_limit_kmh"]) / MAX_SPEED,
            float(raw["truck_game_steer"]),
            float(raw["truck_acceleration_x"]),
            float(raw["truck_acceleration_y"]),
            float(raw["truck_acceleration_z"]),
            float(raw["truck_engine_rpm"]) / MAX_RPM,
            float(raw["truck_displayed_gear"]) / MAX_GEAR,
            float(raw["trailer_attached"]),
            float(raw["trailer_mass_kg"]) / MAX_TRAILER_MASS,
        ],
        dtype=torch.float32,
    )
    return features.unsqueeze(0)


def clamp_prediction(values: np.ndarray) -> Dict[str, float]:
    steering = float(np.clip(values[0], -1.0, 1.0))
    throttle = float(np.clip(values[1], 0.0, 1.0))
    brake = float(np.clip(values[2], 0.0, 1.0))
    return {
        "steering": steering,
        "throttle": throttle,
        "brake": brake,
    }


def apply_ema(pred: Dict[str, float], ema_state: Dict[str, float]) -> Dict[str, float]:
    ema_state["steering"] = EMA_ALPHA * pred["steering"] + (1.0 - EMA_ALPHA) * ema_state["steering"]
    ema_state["throttle"] = EMA_ALPHA * pred["throttle"] + (1.0 - EMA_ALPHA) * ema_state["throttle"]
    ema_state["brake"] = EMA_ALPHA * pred["brake"] + (1.0 - EMA_ALPHA) * ema_state["brake"]
    return {
        "steering": ema_state["steering"],
        "throttle": ema_state["throttle"],
        "brake": ema_state["brake"],
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
    autopilot_enabled: bool,
    pred_raw: Dict[str, float],
    pred_sent: Dict[str, float],
    telemetry: Dict[str, float],
    controller_state: Dict[str, float],
    manual_passthrough: bool,
) -> np.ndarray:
    status = "AUTOPILOT ON" if autopilot_enabled else "AUTOPILOT OFF"
    source = "MODEL" if autopilot_enabled else ("HUMAN->VIRTUAL" if manual_passthrough else "NEUTRAL")

    lines = [
        status,
        f"source | {source}",
        (
            f"pred raw | steer={pred_raw['steering']:+.3f}  "
            f"thr={pred_raw['throttle']:.3f}  "
            f"brk={pred_raw['brake']:.3f}"
        ),
        (
            f"sent     | steer={pred_sent['steering']:+.3f}  "
            f"thr={pred_sent['throttle']:.3f}  "
            f"brk={pred_sent['brake']:.3f}"
        ),
        (
            f"real pad | steer={controller_state['steering']:+.3f}  "
            f"thr={controller_state['throttle']:.3f}  "
            f"brk={controller_state['brake']:.3f}"
        ),
        (
            f"speed | v={telemetry['truck_speed_kmh']:.1f} km/h  "
            f"limit={telemetry['speed_limit_kmh']:.1f} km/h  "
            f"gameSteer={telemetry['truck_game_steer']:+.3f}"
        ),
        (
            f"accel | x={telemetry['truck_acceleration_x']:+.3f}  "
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
        f"Toggle autopilot: controller R button",
        "ESC = exit",
    ]
    return draw_text_lines(frame_bgr, lines)


def make_status_view_image(autopilot_enabled: bool, manual_passthrough: bool) -> np.ndarray:
    frame = np.zeros((STATUS_HEIGHT, STATUS_WIDTH, 3), dtype=np.uint8)

    line1 = "AUTOPILOT ON" if autopilot_enabled else "AUTOPILOT OFF"
    line2 = "VIRTUAL PAD ACTIVE" if autopilot_enabled or manual_passthrough else "VIRTUAL PAD NEUTRAL"

    for idx, text in enumerate([line1, line2]):
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            STATUS_FONT_SCALE,
            STATUS_FONT_THICKNESS,
        )
        text_x = max(10, (STATUS_WIDTH - text_size[0]) // 2)
        text_y = 30 + idx * 30

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            STATUS_FONT_SCALE,
            (255, 255, 255),
            STATUS_FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def safe_destroy_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


# =========================
# Main
# =========================

def main() -> None:
    args = parse_args()
    debug_enabled = args.debug
    manual_passthrough = args.manual_passthrough

    print("[Start] Live inference starting...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"[Setup] Debug window: {'ON' if debug_enabled else 'OFF'}")
    print(f"[Setup] Manual passthrough: {'ON' if manual_passthrough else 'OFF'}")

    print("[Setup] Loading model checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    img_size = int(ckpt["img_size"])
    pretrained = bool(ckpt["pretrained"])

    print(f"[Setup] img_size={img_size}, pretrained={pretrained}")

    print("[Setup] Building model...")
    model = DrivingModel(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print("[Setup] Connecting real controller...")
    controller = SwitchController()
    controller.connect()

    print("[Setup] Connecting telemetry...")
    telemetry_adapter = HttpTelemetryAdapter()
    telemetry_adapter.connect()

    print("[Setup] Creating virtual Xbox controller...")
    virtual_pad = VirtualXboxController()
    virtual_pad.reset()

    autopilot_enabled = False
    toggle_pressed = False
    last_print = 0.0

    ema_state = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    }

    last_sent = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    }

    try:
        with mss() as sct:
            print("[Live] Running. Press controller R button to toggle autopilot. ESC to exit.")

            while True:
                if keyboard.is_pressed(QUIT_KEY):
                    print("[Live] Exit requested")
                    break

                controller_state = controller.read()

                if controller_state["button_toggle"] > 0.5:
                    if not toggle_pressed:
                        autopilot_enabled = not autopilot_enabled
                        print(f"[Live] Autopilot: {'ON' if autopilot_enabled else 'OFF'}")

                        if not autopilot_enabled and not manual_passthrough:
                            virtual_pad.reset()
                            last_sent = {
                                "steering": 0.0,
                                "throttle": 0.0,
                                "brake": 0.0,
                            }

                        toggle_pressed = True
                else:
                    toggle_pressed = False

                image = grab_frame(sct, CAPTURE_REGION)

                raw_telemetry = telemetry_adapter.read().to_dict()
                telemetry = {
                    "truck_speed_kmh": raw_telemetry["truck_speed_kmh"],
                    "speed_limit_kmh": raw_telemetry["speed_limit_kmh"],
                    "truck_game_steer": raw_telemetry["truck_game_steer"],
                    "truck_acceleration_x": raw_telemetry["truck_acceleration_x"],
                    "truck_acceleration_y": raw_telemetry["truck_acceleration_y"],
                    "truck_acceleration_z": raw_telemetry["truck_acceleration_z"],
                    "truck_engine_rpm": raw_telemetry["truck_engine_rpm"],
                    "truck_displayed_gear": raw_telemetry["truck_displayed_gear"],
                    "trailer_attached": raw_telemetry["trailer_attached"],
                    "trailer_mass_kg": raw_telemetry["trailer_mass_kg"],
                }

                image_tensor = transform(image).unsqueeze(0)
                numeric_tensor = build_numeric_features(raw_telemetry)

                with torch.no_grad():
                    output = model(image_tensor, numeric_tensor).squeeze(0).cpu().numpy()

                pred_raw = clamp_prediction(output)
                pred_smooth = apply_ema(pred_raw, ema_state)

                if autopilot_enabled:
                    virtual_pad.apply_controls(
                        steering=pred_smooth["steering"],
                        throttle=pred_smooth["throttle"],
                        brake=pred_smooth["brake"],
                    )
                    last_sent = pred_smooth.copy()
                else:
                    if manual_passthrough:
                        virtual_pad.apply_controls(
                            steering=controller_state["steering"],
                            throttle=controller_state["throttle"],
                            brake=controller_state["brake"],
                        )
                        last_sent = {
                            "steering": float(controller_state["steering"]),
                            "throttle": float(controller_state["throttle"]),
                            "brake": float(controller_state["brake"]),
                        }
                    else:
                        virtual_pad.reset()
                        last_sent = {
                            "steering": 0.0,
                            "throttle": 0.0,
                            "brake": 0.0,
                        }

                status_view_bgr = make_status_view_image(
                    autopilot_enabled=autopilot_enabled,
                    manual_passthrough=manual_passthrough,
                )
                cv2.imshow(MODEL_VIEW_WINDOW, status_view_bgr)

                if debug_enabled:
                    display_image = preprocess_for_display(image)
                    frame_bgr = pil_to_bgr(display_image)
                    frame_bgr = draw_overlay(
                        frame_bgr=frame_bgr,
                        autopilot_enabled=autopilot_enabled,
                        pred_raw=pred_raw,
                        pred_sent=last_sent,
                        telemetry=telemetry,
                        controller_state=controller_state,
                        manual_passthrough=manual_passthrough,
                    )
                    cv2.imshow(DEBUG_VIEW_WINDOW, frame_bgr)
                else:
                    safe_destroy_window(DEBUG_VIEW_WINDOW)

                cv2.waitKey(1)

                now = time.time()
                if now - last_print >= 0.5:
                    source = "MODEL" if autopilot_enabled else ("HUMAN->VIRTUAL" if manual_passthrough else "NEUTRAL")
                    print(
                        f"[Live] auto={autopilot_enabled} src={source} | "
                        f"sent steer={last_sent['steering']:+.3f} "
                        f"thr={last_sent['throttle']:.3f} "
                        f"brk={last_sent['brake']:.3f} | "
                        f"speed={telemetry['truck_speed_kmh']:.1f} "
                        f"limit={telemetry['speed_limit_kmh']:.1f}"
                    )
                    last_print = now

    finally:
        print("[Cleanup] Resetting virtual controller...")
        try:
            virtual_pad.reset()
        except Exception:
            pass
        cv2.destroyAllWindows()
        controller.close()
        telemetry_adapter.close()
        print("[Done] Live inference stopped")


if __name__ == "__main__":
    main()