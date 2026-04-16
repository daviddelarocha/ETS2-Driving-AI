from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import vgamepad as vg

import pygame

# Real controller config (used for toggle + optional manual passthrough)
JOYSTICK_INDEX = 0
CONTROLLER_DEADZONE = 0.08

# Axis mapping for Switch controller
AXIS_STEERING = 0
AXIS_BRAKE = 4
AXIS_THROTTLE = 5

# IMPORTANT: adjust if your R button has another index
BUTTON_TOGGLE_AUTOPILOT = 10


@dataclass
class ControllerState:
    steering: float
    throttle: float
    brake: float
    toggle_autopilot: float

    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, float]:
        return {
            "steering": self.steering,
            "throttle": self.throttle,
            "brake": self.brake,
            "toggle_autopilot": self.toggle_autopilot,
            **self.raw,
        }


class SwitchProControllerAdapter:
    """
    Lector simple para Nintendo Switch Pro Controller usando pygame.joystick.

    Mapeo por defecto:
      axis 0 = stick izquierdo horizontal (steering)
      axis 4 = gatillo izquierdo (brake)
      axis 5 = gatillo derecho (throttle)

    Dependiendo del sistema o tipo de conexión, puede variar.
    """

    def __init__(
        self,
        joystick_index: int = 0,
        deadzone: float = 0.08,
        axis_steering: int = 0,
        axis_brake: int = 4,
        axis_throttle: int = 5,
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
                f"No se encontró mando en índice {self.joystick_index}. "
                f"Mandos detectados: {count}"
            )

        js = pygame.joystick.Joystick(self.joystick_index)
        js.init()
        self.joystick = js

        print("Controller conectado:")
        print(f"  nombre: {js.get_name()}")
        print(f"  axes:   {js.get_numaxes()}")
        print(f"  btns:   {js.get_numbuttons()}")
        print(f"  hats:   {js.get_numhats()}")

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
        """
        Convierte rango típico [-1, 1] a [0, 1].
        """
        out = (value + 1.0) / 2.0
        if out < 0.0:
            return 0.0
        if out > 1.0:
            return 1.0
        return float(out)

    def read(self) -> ControllerState:
        if self.joystick is None:
            raise RuntimeError("El mando no está conectado")

        pygame.event.pump()

        steering_raw = self.joystick.get_axis(self.axis_steering)
        brake_raw = self.joystick.get_axis(self.axis_brake)
        throttle_raw = self.joystick.get_axis(self.axis_throttle)

        steering = self._apply_deadzone(steering_raw)
        brake = self._normalize_trigger(brake_raw)
        throttle = self._normalize_trigger(throttle_raw)

        raw = {
            "axis_steering": float(steering_raw),
            "axis_brake": float(brake_raw),
            "axis_throttle": float(throttle_raw),
        }

        return ControllerState(
            steering=steering,
            throttle=throttle,
            brake=brake,
            raw=raw,
            toggle_autopilot=float(self.joystick.get_button(BUTTON_TOGGLE_AUTOPILOT)),
        )
    
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
