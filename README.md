# Static Gesture CNN

A Python script implementing a 1D CNN for classifying static gestures from landmark data sent via OSC.  
The script supports training, inference, and feature extraction, with a set of OSC commands for interaction.

Useful in conjunction with mediapipe scripts such as: https://github.com/little-scale/mediapipe-js-osc

---

## Features
- **Real-time gesture classification** using a 1D CNN.
- **Train on incoming OSC data** with configurable classes, epochs, and batch size.
- **Output classification results** only above a set confidence threshold.
- **Feature vector output** (`/features`) for use in other systems.
- **Data normalization options** for origin, scale, and rotation.
- **Tweakable model capacity** (number of layers, filters, dense size).
- **Persistent model saving/loading** between sessions.

---

## Parameters (Tweakable Before Runtime)
These are set at the top of the script:

```python
landmark_len = 21             # default landmarks (auto-updates on first input)
class_count = 3               # default number of classes
training_epochs = 50          # default training epochs
batch_size = 32               # default batch size

normalize_origin = True       # subtract first landmark as origin
normalize_scale = True        # scale-normalize by median radius
normalize_rotate = True       # rotate so wrist->middle-MCP lies along +X

CONFIDENCE_THRESHOLD = 0.9    # suppress predictions below this confidence
train_verbose = False         # print per-batch loss if True
infer_verbose = True          # print gesture START/CHANGE/END transitions

EMBED_DIM = 64                 # size of compact /features vector (UDP-safe)

# Model architecture parameters
conv_layers = 2               # number of convolutional layers
filters = 64                  # filters per conv layer
kernel_size = 3               # kernel size for conv layers
dense_units = 128              # dense layer units
dropout_rate = 0.3             # dropout rate
```

---

## OSC Commands
Send these commands to the listening port (default: `9000`).

### Mode Control
- `/mode train` — Switch to training mode (samples will be stored with labels).
- `/mode inference` — Switch to inference mode (incoming data classified in real time).

### Model Management
- `/train` — Train the model with current samples.
- `/train_epochs <int>` — Set the number of epochs before training (default: 50).
- `/reset_model` — Reinitialize the CNN (weights cleared).
- `/clear_data` — Clear all stored training samples.

### Data Input
- `/input <floats> <class_id>` — Input a training or inference sample.  
  Format: **flat list of floats of size `3*n` landmarks (all x, then all y, then all z)**, followed by class ID.

### Parameters at Runtime
- `/set_confidence <float>` — Set the confidence threshold (default: 0.9).
- `/set_classes <int>` — Set the number of output classes.
- `/set_batch_size <int>` — Set the batch size.
- `/set_landmarks <int>` — Set number of landmarks (default: 21).

---

## Outputs
- `/class <int>` — Predicted class index (only if confidence ≥ threshold).
- `/confidence <float>` — Confidence of the predicted class.
- `/features <list>` — Feature embedding vector (size `EMBED_DIM`).

---

## Workflow Example
1. Start the script:  
   ```bash
   python static_gesture_cnn.py
   ```
2. Send `/mode train` to enter training mode.
3. Send samples to `/input` with the format described above.
4. Send `/train` to train the model.
5. Send `/mode inference` to classify new samples.

---

## Tips
- Increase **filters** and **dense_units** for more complex datasets or higher class counts.
- Lower **dropout_rate** for simpler datasets, raise for noisy datasets.
- Normalize inputs (`normalize_origin`, `normalize_scale`, `normalize_rotate`) for better generalization.
- For **3 classes**, a smaller model may suffice (e.g., 2 conv layers, 64 filters).
- For **10+ classes**, increase filters (128–256) and dense units (256–512) for better accuracy.

---

## Requirements
```bash
pip install python-osc tensorflow torch numpy
```
