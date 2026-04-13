# 🚛 ETS2-Autopilot

Deep Learning multimodal autopilot for Euro Truck Simulator 2.

This project implements an **end-to-end autonomous driving system** for Euro Truck Simulator 2 using deep learning.

The model learns to drive by combining:
- 🖼️ Visual input (screen capture)
- 📊 Telemetry data (speed, limits, cargo, etc.)

It predicts continuous control signals and executes them in real time through a **virtual gamepad**, enabling smooth and realistic driving.

---

## 🚀 Features

- **Multimodal architecture**
  - CNN (MobileNetV3) for vision
  - MLP for telemetry
  - Fusion network for control prediction

- **End-to-end learning**
  - Trained from real gameplay data
  - Learns human driving behavior

- **Continuous control outputs**
  - Steering ∈ [-1, 1]
  - Throttle ∈ [0, 1]
  - Brake ∈ [0, 1]

- **Real-time inference**
  - Live screen capture
  - Telemetry integration
  - Low latency predictions

- **Analog control execution**
  - Virtual Xbox controller (no keyboard)
  - Smooth driving behavior
  - No discrete thresholds

---

## 🧠 Training Pipeline

1. **Data Collection**
   - Screen frames
   - Telemetry data
   - Ground truth actions from game physics (`gameSteer`, `gameThrottle`, `gameBrake`)

2. **Dataset Construction**
   - Image + telemetry pairs
   - Continuous targets

3. **Training**
   - Supervised learning
   - Regression on control signals

4. **Evaluation**
   - MAE / RMSE / R²
   - Error percentiles
   - Scatter plots

---

## 🎮 Inference System

The autopilot runs in real time:

- Captures game frames  
- Reads telemetry  
- Predicts control signals  
- Sends them to a **virtual gamepad**

### Controls

- Toggle autopilot using controller button  
- Manual override supported  
- Optional human → virtual controller passthrough  

---

## 📦 Requirements

### Telemetry Server (Required)

Install and run:
https://github.com/Funbit/ets2-telemetry-server

## ⚙️ Usage

### 🧪 Test dataset capture
```bash
python .\collect_dataset.py --test
```

### 📊 Collect dataset
```bash
python .\collect_dataset.py
```

### 🧠 Train model
```bash
python train.py --epochs 10 --batch-size 4 --img-size 160 --num-workers 0
```

### 📈 Evaluate model
```bash
python evaluate.py --batch-size 4 --num-workers 0
```

### 🚗 Run live inference (autopilot)
```bash
python live_inference.py
```