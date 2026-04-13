
# Needed Telemetry server :
https://github.com/Funbit/ets2-telemetry-server#

# To collect the dataset, run the following command:
python .\collect_dataset.py --test

# To train the model, run the following command:
python train.py --epochs 10 --batch-size 4 --img-size 160 --num-workers 0

# To evaluate the model, run the following command:
python evaluate.py --batch-size 4 --num-workers 0

# To run live inference, run the following command:
python live_inference.py
