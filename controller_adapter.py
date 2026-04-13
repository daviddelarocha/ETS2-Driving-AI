from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pygame


@dataclass
class ControllerState:
    steering: float
    throttle: float
    brake: float
    raw: Dict[str, Any]


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
            "axis_steering": steering_raw,
            "axis_brake": brake_raw,
            "axis_throttle": throttle_raw,
        }

        return ControllerState(
            steering=steering,
            throttle=throttle,
            brake=brake,
            raw=raw,
        )