from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


MAX_SPEED = 130.0
MAX_RPM = 3000.0
MAX_GEAR = 12.0
MAX_TRAILER_MASS = 50000.0


class DrivingDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_root: str | Path,
        transform: Callable | None = None,
        verify_images: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_root = Path(images_root)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"No existe el CSV: {self.csv_path}")

        if not self.images_root.exists():
            raise FileNotFoundError(f"No existe la carpeta de imágenes: {self.images_root}")

        print("[DrivingDataset] Reading CSV...")
        df = pd.read_csv(self.csv_path)

        required = {
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
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en CSV: {sorted(missing)}")

        print("[DrivingDataset] Dropping rows with missing values...")
        df = df.dropna(subset=list(required)).reset_index(drop=True)

        if verify_images:
            print("[DrivingDataset] Verifying that image files exist...")
            valid_rows = []
            for _, row in df.iterrows():
                image_path = self.images_root.parent / str(row["image_path"])
                if image_path.exists():
                    valid_rows.append(row)

            df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Dataset vacío.")

        print(f"[DrivingDataset] Final dataset size: {len(df)} samples")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = self.images_root.parent / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        features = torch.tensor(
            [
                float(row["truck_speed_kmh"]) / MAX_SPEED,
                float(row["speed_limit_kmh"]) / MAX_SPEED,
                float(row["truck_game_steer"]),
                float(row["truck_acceleration_x"]),
                float(row["truck_acceleration_y"]),
                float(row["truck_acceleration_z"]),
                float(row["truck_engine_rpm"]) / MAX_RPM,
                float(row["truck_displayed_gear"]) / MAX_GEAR,
                float(row["trailer_attached"]),
                float(row["trailer_mass_kg"]) / MAX_TRAILER_MASS,
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(
            [
                float(row["steering"]),
                float(row["throttle"]),
                float(row["brake"]),
            ],
            dtype=torch.float32,
        )

        return image, features, target