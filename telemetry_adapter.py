from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import requests


@dataclass
class TelemetryState:
    truck_speed_kmh: float
    speed_limit_kmh: float
    truck_game_steer: float

    truck_acceleration_x: float
    truck_acceleration_y: float
    truck_acceleration_z: float

    truck_engine_rpm: float
    truck_displayed_gear: float

    trailer_attached: float
    trailer_mass_kg: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "truck_speed_kmh": self.truck_speed_kmh,
            "speed_limit_kmh": self.speed_limit_kmh,
            "truck_game_steer": self.truck_game_steer,
            "truck_acceleration_x": self.truck_acceleration_x,
            "truck_acceleration_y": self.truck_acceleration_y,
            "truck_acceleration_z": self.truck_acceleration_z,
            "truck_engine_rpm": self.truck_engine_rpm,
            "truck_displayed_gear": self.truck_displayed_gear,
            "trailer_attached": self.trailer_attached,
            "trailer_mass_kg": self.trailer_mass_kg,
        }


class HttpTelemetryAdapter:
    def __init__(self, url: str = "http://127.0.0.1:25555/api/ets2/telemetry") -> None:
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
        navigation = data.get("navigation", {})
        trailer = data.get("trailer", {})

        acceleration = truck.get("acceleration", {})

        attached_raw = trailer.get("attached", False)
        attached = 1.0 if bool(attached_raw) else 0.0

        return TelemetryState(
            truck_speed_kmh=float(truck.get("speed", 0.0)),
            speed_limit_kmh=float(navigation.get("speedLimit", 0.0)),
            truck_game_steer=float(truck.get("gameSteer", 0.0)),
            truck_acceleration_x=float(acceleration.get("x", 0.0)),
            truck_acceleration_y=float(acceleration.get("y", 0.0)),
            truck_acceleration_z=float(acceleration.get("z", 0.0)),
            truck_engine_rpm=float(truck.get("engineRpm", 0.0)),
            truck_displayed_gear=float(truck.get("displayedGear", 0.0)),
            trailer_attached=attached,
            trailer_mass_kg=float(trailer.get("mass", 0.0)),
        )
    