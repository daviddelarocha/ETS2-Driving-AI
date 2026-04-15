import matplotlib.pyplot as plt
from torchvision import transforms
from driving_dataset import DrivingDataset

from train import HideHUD  # reutilizas tu clase

IMG_SIZE = 160

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    HideHUD(IMG_SIZE),
])

dataset = DrivingDataset(
    csv_path="dataset/samples.csv",
    images_root="dataset/images",
    transform=transform,
)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

plt.figure(figsize=(12, 6))

for i in range(8):
    img, _, _ = dataset[i]

    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np * std + mean
    img_np = img_np.clip(0, 1)

    plt.subplot(2, 4, i + 1)
    plt.imshow(img_np)
    plt.axis("off")

plt.show()