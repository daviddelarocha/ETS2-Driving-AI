from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from model import DrivingModel

MAX_SPEED = 130.0
MAX_RPM = 3000.0
MAX_GEAR = 12.0
MAX_TRAILER_MASS = 50000.0

TARGET_NAMES = ["steering", "throttle", "brake"]


class GradCAM:
    def __init__(self, model: DrivingModel, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        numeric_tensor: torch.Tensor,
        target_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(image_tensor, numeric_tensor)
        score = outputs[0, target_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No se capturaron activaciones o gradientes.")

        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()
        if cam_np.max() > 1e-8:
            cam_np = cam_np / cam_np.max()
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        preds_np = outputs.detach().cpu().numpy()[0]
        return cam_np.astype(np.float32), preds_np.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM overlays for all output features"
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Folder containing driving_log.csv and image paths referenced by image_path",
    )
    parser.add_argument(
        "--artifacts-folder",
        type=str,
        required=True,
        help="Folder containing best_model.pt",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of random samples to process",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-csv-name",
        type=str,
        default="samples.csv",
        help="CSV filename inside dataset-folder",
    )
    return parser.parse_args()

def build_numeric_tensor(
    truck_speed_kmh: float,
    speed_limit_kmh: float,
    truck_game_steer: float,
    acc_x: float,
    acc_y: float,
    acc_z: float,
    rpm: float,
    gear: float,
    trailer_attached: float,
    trailer_mass: float,
) -> torch.Tensor:
    features = torch.tensor(
        [
            truck_speed_kmh / MAX_SPEED,
            speed_limit_kmh / MAX_SPEED,
            truck_game_steer,
            acc_x,
            acc_y,
            acc_z,
            rpm / MAX_RPM,
            gear / MAX_GEAR,
            trailer_attached,
            trailer_mass / MAX_TRAILER_MASS,
        ],
        dtype=torch.float32,
    )
    return features.unsqueeze(0)


def clamp_prediction(values: np.ndarray) -> Dict[str, float]:
    return {
        "steering": float(np.clip(values[0], -1.0, 1.0)),
        "throttle": float(np.clip(values[1], 0.0, 1.0)),
        "brake": float(np.clip(values[2], 0.0, 1.0)),
    }


def load_model_and_transform(model_path: Path) -> Tuple[DrivingModel, transforms.Compose, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    img_size = int(checkpoint["img_size"])
    pretrained = bool(checkpoint["pretrained"])

    model = DrivingModel(pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform, img_size


def load_image(image_path: Path) -> Tuple[Image.Image, np.ndarray]:
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    rgb_np = np.array(pil_img)
    return pil_img, rgb_np


def resize_cam_to_image(cam: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    return cam_resized


def make_heatmap(cam_resized: np.ndarray) -> np.ndarray:
    cam_uint8 = np.uint8(cam_resized * 255.0)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_heatmap(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(
        image_rgb.astype(np.uint8),
        1.0 - alpha,
        heatmap_rgb.astype(np.uint8),
        alpha,
        0.0,
    )


def format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.3f}"


def add_text_block(
    image_rgb: np.ndarray,
    sample_name: str,
    target_name: str,
    pred: Dict[str, float],
    telemetry: Dict[str, float],
    dataset_targets: Dict[str, float | None],
) -> np.ndarray:
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    lines = [
        f"sample={sample_name}",
        f"gradcam_target={target_name}",
        "",
        "INPUTS / FEATURES",
        f"speed={telemetry['truck_speed_kmh']:.1f} km/h",
        f"limit={telemetry['speed_limit_kmh']:.1f} km/h",
        f"gameSteer={telemetry['truck_game_steer']:+.3f}",
        f"acc_x={telemetry['truck_acceleration_x']:+.3f}",
        f"acc_y={telemetry['truck_acceleration_y']:+.3f}",
        f"acc_z={telemetry['truck_acceleration_z']:+.3f}",
        f"rpm={telemetry['truck_engine_rpm']:.0f}",
        f"gear={telemetry['truck_displayed_gear']:.0f}",
        f"trailer_attached={int(telemetry['trailer_attached'])}",
        f"trailer_mass={telemetry['trailer_mass_kg']:.0f} kg",
        "",
        "REAL TARGETS",
        f"real steering={format_value(dataset_targets.get('steering'))}",
        f"real throttle={format_value(dataset_targets.get('throttle'))}",
        f"real brake={format_value(dataset_targets.get('brake'))}",
        "",
        "MODEL OUTPUTS",
        f"pred steering={pred['steering']:+.3f}",
        f"pred throttle={pred['throttle']:.3f}",
        f"pred brake={pred['brake']:.3f}",
    ]

    y = 26
    for line in lines:
        if line == "":
            y += 10
            continue

        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 20

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def save_outputs(
    output_dir: Path,
    sample_name: str,
    target_name: str,
    overlay_rgb: np.ndarray,
    metadata: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base = f"{sample_name}_{target_name}"
    Image.fromarray(overlay_rgb).save(output_dir / f"{base}_overlay.png")
    # (output_dir / f"{base}_info.json").write_text(
    #     json.dumps(metadata, indent=2),
    #     encoding="utf-8",
    # )


def process_single_sample(
    model: DrivingModel,
    transform: transforms.Compose,
    gradcam: GradCAM,
    image_path: Path,
    telemetry: Dict[str, float],
    dataset_targets: Dict[str, float | None],
    target_name: str,
    output_dir: Path,
    overlay_alpha: float,
    sample_name: str,
) -> Dict[str, float]:
    pil_img, original_rgb = load_image(image_path)

    image_tensor = transform(pil_img).unsqueeze(0)
    numeric_tensor = build_numeric_tensor(
        truck_speed_kmh=telemetry["truck_speed_kmh"],
        speed_limit_kmh=telemetry["speed_limit_kmh"],
        truck_game_steer=telemetry["truck_game_steer"],
        acc_x=telemetry["truck_acceleration_x"],
        acc_y=telemetry["truck_acceleration_y"],
        acc_z=telemetry["truck_acceleration_z"],
        rpm=telemetry["truck_engine_rpm"],
        gear=telemetry["truck_displayed_gear"],
        trailer_attached=telemetry["trailer_attached"],
        trailer_mass=telemetry["trailer_mass_kg"],
    )

    target_index = TARGET_NAMES.index(target_name)
    cam, raw_pred = gradcam.generate(
        image_tensor=image_tensor,
        numeric_tensor=numeric_tensor,
        target_index=target_index,
    )

    pred = clamp_prediction(raw_pred)

    cam_resized = resize_cam_to_image(cam, original_rgb.shape)
    heatmap_rgb = make_heatmap(cam_resized)
    overlay_rgb = overlay_heatmap(original_rgb, heatmap_rgb, overlay_alpha)
    overlay_rgb = add_text_block(
        overlay_rgb,
        sample_name=sample_name,
        target_name=target_name,
        pred=pred,
        telemetry=telemetry,
        dataset_targets=dataset_targets,
    )

    metadata = {
        "sample_name": sample_name,
        "image_path": str(image_path),
        "gradcam_target": target_name,
        "inputs": telemetry,
        "real_targets": dataset_targets,
        "model_outputs": pred,
    }

    save_outputs(
        output_dir=output_dir,
        sample_name=sample_name,
        target_name=target_name,
        overlay_rgb=overlay_rgb,
        metadata=metadata,
    )

    return pred


def load_dataset_rows(csv_path: Path, dataset_folder: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")
    if not dataset_folder.exists():
        raise FileNotFoundError(f"No existe la carpeta dataset: {dataset_folder}")

    df = pd.read_csv(csv_path)

    required_columns = {
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
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

    valid_rows = []
    resolved_paths = []

    for _, row in df.iterrows():
        rel_path = Path(str(row["image_path"]))
        img_path = rel_path if rel_path.is_absolute() else dataset_folder / rel_path
        if img_path.exists():
            valid_rows.append(True)
            resolved_paths.append(str(img_path.resolve()))
        else:
            valid_rows.append(False)

    df = df.loc[valid_rows].copy().reset_index(drop=True)
    df["resolved_image_path"] = resolved_paths

    if len(df) == 0:
        raise ValueError("No hay filas válidas con imagen existente.")

    return df


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_folder = Path(args.dataset_folder)
    artifacts_folder = Path(args.artifacts_folder)

    model_path = artifacts_folder / "best_model.pt"
    output_dir = artifacts_folder / "gradcam"
    csv_path = dataset_folder / args.dataset_csv_name

    model, transform, img_size = load_model_and_transform(model_path)
    target_layer = model.image_backbone[-1]
    gradcam = GradCAM(model, target_layer)

    try:
        df = load_dataset_rows(csv_path, dataset_folder)

        n = min(args.num_samples, len(df))
        selected_indices = random.sample(range(len(df)), n)

        print(f"[INFO] Dataset samples disponibles: {len(df)}")
        print(f"[INFO] Seleccionando {n} muestras aleatorias")
        print(f"[INFO] Checkpoint img_size: {img_size}")
        print(f"[INFO] Targets a generar: {TARGET_NAMES}")

        summary = []

        for i, idx in enumerate(selected_indices, start=1):
            row = df.iloc[idx]

            image_path = Path(str(row["resolved_image_path"]))
            sample_name = Path(str(row["image_path"])).stem

            telemetry = {
                "truck_speed_kmh": float(row["truck_speed_kmh"]),
                "speed_limit_kmh": float(row["speed_limit_kmh"]),
                "truck_game_steer": float(row["truck_game_steer"]),
                "truck_acceleration_x": float(row["truck_acceleration_x"]),
                "truck_acceleration_y": float(row["truck_acceleration_y"]),
                "truck_acceleration_z": float(row["truck_acceleration_z"]),
                "truck_engine_rpm": float(row["truck_engine_rpm"]),
                "truck_displayed_gear": float(row["truck_displayed_gear"]),
                "trailer_attached": float(row["trailer_attached"]),
                "trailer_mass_kg": float(row["trailer_mass_kg"]),
            }

            dataset_targets = {
                "steering": float(row["steering"]),
                "throttle": float(row["throttle"]),
                "brake": float(row["brake"]),
            }

            per_target_preds = {}

            for target_name in TARGET_NAMES:
                pred = process_single_sample(
                    model=model,
                    transform=transform,
                    gradcam=gradcam,
                    image_path=image_path,
                    telemetry=telemetry,
                    dataset_targets=dataset_targets,
                    target_name=target_name,
                    output_dir=output_dir,
                    overlay_alpha=args.overlay_alpha,
                    sample_name=sample_name,
                )
                per_target_preds[target_name] = pred

            summary.append({
                "sample_name": sample_name,
                "image_path": str(image_path),
                "inputs": telemetry,
                "real_targets": dataset_targets,
                "generated_targets": TARGET_NAMES,
                "predictions_per_gradcam_target": per_target_preds,
            })

            print(f"[{i}/{n}] OK -> {sample_name}")

        (output_dir / "summary_all_targets.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(f"[OK] Procesadas {n} muestras")
        print(f"[OK] Salida guardada en: {output_dir.resolve()}")

    finally:
        gradcam.close()


if __name__ == "__main__":
    main()
