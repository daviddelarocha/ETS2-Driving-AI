import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import functional as F

from driving_dataset import DrivingDataset
from train import HideHUD

IMG_SIZE = 160

transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.CenterCrop((80, 220)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # ),
])

dataset = DrivingDataset(
    csv_path="dataset/samples.csv",
    images_root="dataset/images",
    transform=transform,
)

plt.figure(figsize=(12, 6))

for j in range(8):
    # pick random index
    i = np.random.randint(0, len(dataset))
    img, _, _ = dataset[i]

    # img_np = img.permute(1, 2, 0).numpy()
    # img_np = img_np * std + mean
    # img_np = img_np.clip(0, 1)

    plt.subplot(2, 4, j + 1)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.axis("off")

plt.show()