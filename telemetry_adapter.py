from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import requests


@dataclass
class TelemetryState:
    truck_speed_kmh: float
    truck_accel_kmh_s: float
    speed_limit_kmh: float

    user_steer: float
    user_throttle: float
    user_brake: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "truck_speed_kmh": self.truck_speed_kmh,
            "truck_accel_kmh_s": self.truck_accel_kmh_s,
            "speed_limit_kmh": self.speed_limit_kmh,
            "user_steer": self.user_steer,
            "user_throttle": self.user_throttle,
            "user_brake": self.user_brake,
        }


class HttpTelemetryAdapter:
    def __init__(self, url: str = "http://127.0.0.1:25555/api/ets2/telemetry") -> None:
        self.url = url
        self.session = requests.Session()
        self._last_speed_kmh: float | None = None
        self._last_time: float | None = None

    def connect(self) -> None:
        r = self.session.get(self.url, timeout=2)
        r.raise_for_status()

    def close(self) -> None:
        self.session.close()

    def read(self) -> TelemetryState:
        r = self.session.get(self.url, timeout=1)
        r.raise_for_status()
        data = r.json()

        truck = data.get("truck", {})
        navigation = data.get("navigation", {})

        now = time.perf_counter()
        speed_kmh = float(truck.get("speed", 0.0))

        accel_kmh_s = 0.0
        if self._last_speed_kmh is not None and self._last_time is not None:
            dt = now - self._last_time
            if dt > 1e-6:
                accel_kmh_s = (speed_kmh - self._last_speed_kmh) / dt

        self._last_speed_kmh = speed_kmh
        self._last_time = now

        return TelemetryState(
            truck_speed_kmh=speed_kmh,
            truck_accel_kmh_s=accel_kmh_s,
            speed_limit_kmh=float(navigation.get("speedLimit", 0.0)),
            user_steer=float(truck.get("userSteer", 0.0)),
            user_throttle=float(truck.get("userThrottle", 0.0)),
            user_brake=float(truck.get("userBrake", 0.0)),
        )
    