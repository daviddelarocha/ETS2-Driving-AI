from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import cv2
import keyboard
import numpy as np
import pygame
import torch
import torch.nn as nn
import vgamepad as vg
from mss import mss
from PIL import Image
from torchvision import models, transforms

from telemetry_adapter import HttpTelemetryAdapter


# =========================
# Config
# =========================

MODEL_PATH = Path("artifacts_puns/best_model.pt")
TRUCK_CONFIG_PATH = Path("truck_config.json")

CAPTURE_MONITOR = 1
CAPTURE_REGION = None

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

QUIT_KEY = "esc"

# Real controller config (used for toggle + optional manual passthrough)
JOYSTICK_INDEX = 0
CONTROLLER_DEADZONE = 0.08

# Axis mapping for Switch controller
AXIS_STEERING = 0
AXIS_BRAKE = 4
AXIS_THROTTLE = 5

# IMPORTANT: adjust if your R button has another index
BUTTON_TOGGLE_AUTOPILOT = 10

# Telemetry normalization
MAX_SPEED_KMH = 130.0
MAX_CARGO_MASS_KG = 50000.0
MAX_POWER_HP = 1000.0

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
# Model
# =========================

class DrivingModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.image_backbone = backbone.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)

        image_feature_dim = 576
        numeric_feature_dim = 4

        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        x_img = self.image_backbone(image)
        x_img = self.image_pool(x_img)
        x_img = torch.flatten(x_img, 1)

        x_num = self.numeric_mlp(numeric)

        x = torch.cat([x_img, x_num], dim=1)
        return self.head(x)


# =========================
# Real controller reader
# =========================

class SwitchController:
    def __init__(
        self,
        joystick_index: int = JOYSTICK_INDEX,
        deadzone: float = CONTROLLER_DEADZONE,
        axis_steering: int = AXIS_STEERING,
        axis_brake: int = AXIS_BRAKE,
        axis_throttle: int = AXIS_THROTTLE,
    ) -> None:
        self.joystick_index = joystick_index
        self.deadzone = deadzone
        self.axis_steering = axis_steering
        self.axis_brake = axis_brake
        self.axis_throttle = axis_throttle
        self.joystick: pygame.joystick.Joystick | None = None

    def connect(self) -> None:
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count <= self.joystick_index:
            raise RuntimeError(
                f"No controller found at index {self.joystick_index}. Controllers detected: {count}"
            )

        js = pygame.joystick.Joystick(self.joystick_index)
        js.init()
        self.joystick = js

        print("[Controller] Connected")
        print(f"[Controller] name={js.get_name()}")
        print(f"[Controller] axes={js.get_numaxes()}")
        print(f"[Controller] buttons={js.get_numbuttons()}")
        print(f"[Controller] hats={js.get_numhats()}")

    def close(self) -> None:
        try:
            if self.joystick is not None:
                self.joystick.quit()
        finally:
            pygame.joystick.quit()
            pygame.quit()

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return float(value)

    def _normalize_trigger(self, value: float) -> float:
        out = (value + 1.0) / 2.0
        return max(0.0, min(1.0, float(out)))

    def read(self) -> Dict[str, float]:
        if self.joystick is None:
            raise RuntimeError("Controller not connected")

        pygame.event.pump()

        steering_raw = self.joystick.get_axis(self.axis_steering)
        brake_raw = self.joystick.get_axis(self.axis_brake)
        throttle_raw = self.joystick.get_axis(self.axis_throttle)

        return {
            "steering": self._apply_deadzone(steering_raw),
            "brake": self._normalize_trigger(brake_raw),
            "throttle": self._normalize_trigger(throttle_raw),
            "button_toggle": float(self.joystick.get_button(BUTTON_TOGGLE_AUTOPILOT)),
        }


# =========================
# Virtual controller output
# =========================

class VirtualXboxController:
    def __init__(self) -> None:
        self.gamepad = vg.VX360Gamepad()

    def reset(self) -> None:
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.update()

    def apply_controls(self, steering: float, throttle: float, brake: float) -> None:
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        self.gamepad.left_joystick_float(
            x_value_float=steering,
            y_value_float=0.0,
        )
        self.gamepad.right_trigger_float(value_float=throttle)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.update()


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


def load_truck_config(path: Path = TRUCK_CONFIG_PATH) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "truck_power_hp": float(data["truck_power_hp"]),
    }


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


def build_numeric_features(raw_telemetry: Dict[str, float], truck_cfg: Dict[str, float]) -> torch.Tensor:
    features = torch.tensor(
        [
            float(raw_telemetry["truck_speed_kmh"]) / MAX_SPEED_KMH,
            float(raw_telemetry["speed_limit_kmh"]) / MAX_SPEED_KMH,
            float(raw_telemetry["cargo_mass_kg"]) / MAX_CARGO_MASS_KG,
            float(truck_cfg["truck_power_hp"]) / MAX_POWER_HP,
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
            f"telemetry | speed={telemetry['truck_speed_kmh']:.1f} km/h  "
            f"limit={telemetry['speed_limit_kmh']:.1f} km/h"
        ),
        (
            f"truck | cargo={telemetry['cargo_mass_kg']:.0f} kg  "
            f"power={telemetry['truck_power_hp']:.0f} hp"
        ),
        f"Toggle autopilot: controller R button (index {BUTTON_TOGGLE_AUTOPILOT})",
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

    if not TRUCK_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Truck config not found: {TRUCK_CONFIG_PATH}")

    print(f"[Setup] Debug window: {'ON' if debug_enabled else 'OFF'}")
    print(f"[Setup] Manual passthrough: {'ON' if manual_passthrough else 'OFF'}")

    print("[Setup] Loading truck config...")
    truck_cfg = load_truck_config()

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
                    "cargo_mass_kg": raw_telemetry["cargo_mass_kg"],
                    "truck_power_hp": truck_cfg["truck_power_hp"],
                }

                image_tensor = transform(image).unsqueeze(0)
                numeric_tensor = build_numeric_features(raw_telemetry, truck_cfg)

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