from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
            "cargo_mass_kg",
            "truck_power_hp",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en CSV: {sorted(missing)}")

        print("[DrivingDataset] Dropping rows with missing values...")
        df = df.dropna(
            subset=[
                "image_path",
                "steering",
                "throttle",
                "brake",
                "truck_speed_kmh",
                "speed_limit_kmh",
                "cargo_mass_kg",
                "truck_power_hp",
            ]
        ).reset_index(drop=True)

        if verify_images:
            print("[DrivingDataset] Verifying that image files exist...")
            valid_rows = []
            missing_count = 0

            for _, row in df.iterrows():
                image_rel = str(row["image_path"])
                image_path = self.images_root.parent / image_rel

                if image_path.exists():
                    valid_rows.append(row)
                else:
                    missing_count += 1

            df = pd.DataFrame(valid_rows).reset_index(drop=True)

            print(f"[DrivingDataset] Valid images: {len(df)}")
            print(f"[DrivingDataset] Missing images ignored: {missing_count}")

        self.df = df

        if len(self.df) == 0:
            raise ValueError("Dataset vacío después de filtrar imágenes.")

        print(f"[DrivingDataset] Final dataset size: {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_rel = str(row["image_path"])
        image_path = self.images_root.parent / image_rel

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        features = torch.tensor(
            [
                float(row["truck_speed_kmh"]) / 130.0,
                float(row["speed_limit_kmh"]) / 130.0,
                float(row["cargo_mass_kg"]) / 50000.0,
                float(row["truck_power_hp"]) / 1000.0,
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