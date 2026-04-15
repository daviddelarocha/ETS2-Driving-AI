import torch.nn as nn
import torch
from torchvision import models


class DrivingModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        print(f"[Model] Building MobileNetV3 Small backbone | pretrained={pretrained}")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.image_backbone = backbone.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)

        image_feature_dim = 576
        numeric_feature_dim = 10

        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 128),
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
    
    
class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weights: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_output_loss = torch.nn.functional.smooth_l1_loss(
            pred,
            target,
            reduction="none",
        )
        weighted = per_output_loss * self.weights
        return weighted.mean()
