# Static Gesture CNN
## Gesture Classifier — OSC-Controlled 1D CNN

This Python script implements a real-time hand gesture classifier using a 1D Convolutional Neural Network (CNN). It communicates via [OSC (Open Sound Control)](https://opensoundcontrol.stanford.edu/) and is designed for use with real-time creative environments like Max, SuperCollider, TouchDesigner, etc.

---

## Features

- Receives hand pose data (x, y, z coordinates) via OSC
- Supports training and inference modes
- Dynamically configurable number of classes, epochs, and landmarks
- Origin normalization (relative to first landmark)
- Confidence threshold for prediction output
- Model saving and loading
- Feature vector output for visualization

---

## OSC Message Format

### `/input`

Send one OSC message containing a **flat list of floats** with this structure:

```
/input x0 x1 ... xn y0 y1 ... yn z0 z1 ... zn class_id
```

- `x`, `y`, `z` = coordinate lists of equal length
- `n` = number of landmarks (default: 21)
- `class_id` = zero-indexed class integer (ignored during inference)

Example: For 21 landmarks (3 × 21 + 1 = 64 floats total):
```
/input 0.1 0.2 ... 0.9 0.1 0.2 ... 0.9 0.1 0.2 ... 0.9 1
```

---

## Modes

Use `/set_mode` to switch between modes:

- `train`: stores labeled samples for training
- `inference`: runs predictions

### `/set_mode`

```
/set_mode train
/set_mode inference
```

---

## ⚙Configuration Commands

### `/set_classes <int>`
Set the number of output classes (e.g. 3 for a 3-class classifier)

```
/set_classes 3
```

### `/set_epochs <int>`
Set the number of training epochs (default: 100)

```
/set_epochs 200
```

### `/set_confidence_threshold <float>`
Set the minimum confidence required to output a class prediction (default: 0.7)

```
/set_confidence_threshold 0.65
```

### `/normalize <0 or 1>`
Toggle origin normalization (default: 1 = ON). When ON, all landmarks are made relative to the first.

```
/normalize 1
```

---

## Training Control

### `/train`
Train the model using all stored samples.

```
/train
```

### `/clear_training`
Clear all stored training data (does not reset the model weights).

```
/clear_training
```

### `/reset_model`
Reset model weights (preserves current class count and landmark count).

```
/reset_model
```

---

## Model Persistence

### `/save_model`
Save the trained model to `gesture_model.pt`.

```
/save_model
```

### `/load_model`
Load model from `gesture_model.pt`.

```
/load_model
```

---

## Inference Output

When in `inference` mode, the script outputs:

- `/confidence <float>`  
  The highest softmax probability from the classifier

- `/predicted_class <int>`  
  Only sent if confidence ≥ `CONFIDENCE_THRESHOLD`

- `/features [float list]`  
  A flattened vector from the last CNN layer (useful for visualization)

---

## Typical Workflow

1. Start in `train` mode:
    ```
    /set_mode train
    ```
2. Send multiple labeled samples via `/input`
3. Train:
    ```
    /train
    ```
4. Switch to inference mode:
    ```
    /set_mode inference
    ```
5. Send new samples and receive predictions if confidence is high enough.

---

## Dependencies

Install via pip:

```bash
pip install torch python-osc
```

---

## Questions or Extensions?

This tool is meant to be adapted to your real-time creative workflow. You can easily extend it with:

- Custom gesture sets
- Real-time Max patch control
- Multimodal input (e.g., face + hands)
- Confusion matrix output
- Unknown gesture detection

Let me know if you'd like a Max patch or demo scripts!
