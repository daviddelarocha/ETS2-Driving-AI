import lmdb
import pickle
from pathlib import Path
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm


def pack_dataset(csv_path, images_root, output_path):
    csv_path = Path(csv_path)
    images_root = Path(images_root)

    df = pd.read_csv(csv_path)

    env = lmdb.open(
        str(output_path),
        map_size=1024**4,  # 1TB virtual
        subdir=True,
        meminit=False,
        map_async=True,
    )

    with env.begin(write=True) as txn:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_path = images_root / row["image_path"].replace("images/", "")
            if not img_path.exists():
                continue

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            sample = {
                "image": img_bytes,
                "features": {
                    "truck_speed_kmh": row["truck_speed_kmh"],
                    "speed_limit_kmh": row["speed_limit_kmh"],
                    "truck_game_steer": row["truck_game_steer"],
                    "truck_acceleration_x": row["truck_acceleration_x"],
                    "truck_acceleration_y": row["truck_acceleration_y"],
                    "truck_acceleration_z": row["truck_acceleration_z"],
                    "truck_engine_rpm": row["truck_engine_rpm"],
                    "truck_displayed_gear": row["truck_displayed_gear"],
                    "trailer_attached": row["trailer_attached"],
                    "trailer_mass_kg": row["trailer_mass_kg"],
                },
                "target": {
                    "steering": row["steering"],
                    "throttle": row["throttle"],
                    "brake": row["brake"],
                },
            }

            key = f"{idx:08d}".encode()
            txn.put(key, pickle.dumps(sample))

        txn.put(b"length", pickle.dumps(len(df)))

    env.sync()
    env.close()
    print("LMDB creado en:", output_path)


if __name__ == "__main__":
    pack_dataset(
        csv_path="dataset/samples.csv",
        images_root="dataset/images",
        output_path="dataset.lmdb",
    )