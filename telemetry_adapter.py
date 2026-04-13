from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import requests


@dataclass
class TelemetryState:
    truck_speed_kmh: float
    speed_limit_kmh: float
    cargo_mass_kg: float
    game_steer: float
    game_throttle: float
    game_brake: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "truck_speed_kmh": self.truck_speed_kmh,
            "speed_limit_kmh": self.speed_limit_kmh,
            "cargo_mass_kg": self.cargo_mass_kg,
            "game_steer": self.game_steer,
            "game_throttle": self.game_throttle,
            "game_brake": self.game_brake,
        }


class HttpTelemetryAdapter:
    def __init__(self, url: str = "http://192.168.1.48:25555/api/ets2/telemetry") -> None:
        self.url = url
        self.session = requests.Session()

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
        trailer = data.get("trailer", {})
        navigation = data.get("navigation", {})

        return TelemetryState(
            truck_speed_kmh=float(truck.get("speed", 0.0)),
            speed_limit_kmh=float(navigation.get("speedLimit", 0.0)),
            cargo_mass_kg=float(trailer.get("mass", 0.0)),
            game_steer=float(truck.get("gameSteer", 0.0)),
            game_throttle=float(truck.get("gameThrottle", 0.0)),
            game_brake=float(truck.get("gameBrake", 0.0)),
        )