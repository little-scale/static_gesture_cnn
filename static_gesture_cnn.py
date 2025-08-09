import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pythonosc import dispatcher, osc_server, udp_client

# =========================
# CONFIG
# =========================
OSC_RECEIVE_PORT = 9000
OSC_SEND_PORT = 9001
OSC_SEND_IP = "127.0.0.1"
MODEL_SAVE_PATH = "gesture_model.pt"

# =========================
# PARAMETERS (tweakable)
# =========================
landmark_len = 21            # default landmarks (auto-updates on first input)
class_count   = 5            # default number of classes
training_epochs = 100         # default epochs (you set to 50)
batch_size      = 32         # default batch size

normalize_origin = True      # subtract first landmark as origin
normalize_scale  = True      # scale-normalize by median radius
normalize_rotate = True      # rotate so wrist->middle-MCP lies along +X

CONFIDENCE_THRESHOLD = 0.9   # suppress predictions below this (you set to 0.9)
train_verbose = False        # print per-batch loss if True
infer_verbose = True         # print gesture START/CHANGE/END transitions

# ===== MODEL CAPACITY (pre-run, no OSC) =====
CHANNELS      = [64, 128, 128]  # width & depth (per Conv1d block)
KERNEL_SIZE   = 3               # 3 or 5 usually best
USE_BATCHNORM = True
DROPOUT_P     = 0.2             # 0.0–0.3; raise if you add capacity
EMBED_DIM     = 64              # /features length (UDP-safe)

# =========================
# STATE
# =========================
data_buffer = []             # list of [1, 3, n] tensors
label_buffer = []            # list of ints
current_mode = "inference"   # "train" or "inference"
current_active_class = None  # for inference transition logs

model = None
optimizer = None
loss_fn = nn.CrossEntropyLoss()
client = udp_client.SimpleUDPClient(OSC_SEND_IP, OSC_SEND_PORT)

# =========================
# MODEL
# =========================
class GestureCNN(nn.Module):
    """
    Configurable 1D CNN:
      - Conv blocks defined by CHANNELS
      - Global average pooling
      - Fixed-size embedding for /features (EMBED_DIM)
    """
    def __init__(self, input_len: int, num_classes: int):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in CHANNELS:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2))
            if USE_BATCHNORM:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if DROPOUT_P > 0:
                layers.append(nn.Dropout(DROPOUT_P))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.pool  = nn.AdaptiveAvgPool1d(1)   # [B, C, 1] regardless of landmark_len
        self.embed = nn.Linear(in_ch, EMBED_DIM)
        self.head  = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: [B, 3, N]
        x = self.backbone(x)                        # [B, C, N]
        pooled = self.pool(x).squeeze(2)            # [B, C]
        feat_embed = torch.tanh(self.embed(pooled)) # [B, EMBED_DIM]
        logits = self.head(pooled)                  # [B, num_classes]
        return logits, feat_embed

def init_model(n_landmarks: int, n_classes: int):
    global model, optimizer, landmark_len, class_count
    landmark_len = n_landmarks
    class_count = n_classes
    model = GestureCNN(landmark_len, class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"> Model initialized with {n_landmarks} landmarks and {n_classes} classes.")

# =========================
# NORMALIZATION HELPERS
# =========================
def apply_normalizations(landmarks):
    """
    landmarks: list[[x,y,z], ...] AFTER optional origin subtraction.
    Applies:
      - scale normalization by median radius (if enabled)
      - rotation so vector (0->9) lies along +X (if enabled & indices present)
    Returns: list[[x,y,z], ...]
    """
    # Scale by median radius from origin
    if normalize_scale:
        radii = [math.sqrt(x*x + y*y + z*z) for x, y, z in landmarks]
        if radii:
            med = sorted(radii)[len(radii)//2]
            scale = med if med > 1e-8 else 1.0
            landmarks = [[x/scale, y/scale, z/scale] for x, y, z in landmarks]

    # Rotate so wrist->middle-MCP (0 -> 9) points along +X (MediaPipe Hands index)
    if normalize_rotate and len(landmarks) >= 10:
        vx, vy = landmarks[9][0], landmarks[9][1]
        if abs(vx) + abs(vy) > 1e-8:
            ang = math.atan2(vy, vx)
            cosA, sinA = math.cos(-ang), math.sin(-ang)
            rotated = []
            for x, y, z in landmarks:
                xr = x*cosA - y*sinA
                yr = x*sinA + y*cosA
                rotated.append([xr, yr, z])
            landmarks = rotated

    return landmarks

# =========================
# OSC HANDLERS
# =========================
def set_mode(addr, *args):
    """Accept 'train'/'training' or 'infer'/'inference'."""
    global current_mode, current_active_class
    if len(args) < 1:
        print("> Error: /set_mode requires an argument")
        return
    val = str(args[0]).lower()
    aliases = {"train":"train", "training":"train", "infer":"inference", "inference":"inference"}
    if val in aliases:
        current_mode = aliases[val]
        current_active_class = None  # reset active state on mode change
        print(f"> Mode set to: {current_mode}")
    else:
        print(f"! Invalid mode: {args[0]} (use train/training or infer/inference)")

def set_class_count(addr, *args):
    if len(args) < 1:
        print("> Error: /set_classes requires an integer")
        return
    init_model(landmark_len, int(args[0]))

def set_normalization(addr, *args):
    global normalize_origin
    if len(args) < 1:
        print("> Error: /normalize requires 0 or 1")
        return
    normalize_origin = bool(int(args[0]))
    print(f"> Normalize origin set to: {normalize_origin}")

def set_normalize_scale(addr, *args):
    global normalize_scale
    if len(args) < 1:
        print("> Error: /normalize_scale requires 0 or 1")
        return
    normalize_scale = bool(int(args[0]))
    print(f"> Normalize scale set to: {normalize_scale}")

def set_normalize_rotate(addr, *args):
    global normalize_rotate
    if len(args) < 1:
        print("> Error: /normalize_rotate requires 0 or 1")
        return
    normalize_rotate = bool(int(args[0]))
    print(f"> Normalize rotate set to: {normalize_rotate}")

def set_epochs(addr, *args):
    global training_epochs
    if len(args) < 1:
        print("> Error: /set_epochs requires an integer")
        return
    training_epochs = int(args[0])
    print(f"> Training epochs set to: {training_epochs}")

def set_batch_size(addr, *args):
    global batch_size
    if len(args) < 1:
        print("> Error: /set_batch_size requires an integer")
        return
    batch_size = max(1, int(args[0]))
    print(f"> Batch size set to: {batch_size}")

def set_confidence_threshold(addr, *args):
    global CONFIDENCE_THRESHOLD
    if len(args) < 1:
        print("> Error: /set_confidence_threshold requires a float")
        return
    CONFIDENCE_THRESHOLD = float(args[0])
    print(f"> Confidence threshold set to: {CONFIDENCE_THRESHOLD:.2f}")

def set_train_verbose(addr, *args):
    global train_verbose
    if len(args) < 1:
        print("> Error: /set_train_verbose requires 0 or 1")
        return
    train_verbose = bool(int(args[0]))
    print(f"> Train verbose set to: {train_verbose}")

def set_infer_verbose(addr, *args):
    global infer_verbose
    if len(args) < 1:
        print("> Error: /set_infer_verbose requires 0 or 1")
        return
    infer_verbose = bool(int(args[0]))
    print(f"> Inference verbose set to: {infer_verbose}")

def reset_model(addr, *args):
    init_model(landmark_len, class_count)
    print("> Model reset to initial weights.")

def clear_training(addr, *args):
    global data_buffer, label_buffer
    data_buffer = []
    label_buffer = []
    print("> Cleared all stored training data.")

def receive_sample(addr, *args):
    """
    /input x0..xN-1 y0..yN-1 z0..zN-1 class_id
    """
    global model, current_active_class
    if model is None:
        print("Model not initialized.")
        return

    if len(args) < 4 or (len(args) - 1) % 3 != 0:
        print("> Error: /input requires 3*N floats followed by 1 class int.")
        return

    num_landmarks = (len(args) - 1) // 3
    class_id = int(args[-1])

    x_vals = args[0:num_landmarks]
    y_vals = args[num_landmarks:num_landmarks * 2]
    z_vals = args[num_landmarks * 2:num_landmarks * 3]

    if num_landmarks != landmark_len:
        init_model(num_landmarks, class_count)

    # Build landmarks [[x,y,z], ...]
    landmarks = list(zip(x_vals, y_vals, z_vals))

    # Origin normalize
    if normalize_origin:
        ox, oy, oz = landmarks[0]
        landmarks = [[x - ox, y - oy, z - oz] for x, y, z in landmarks]

    # Scale + rotate normalize
    landmarks = apply_normalizations(landmarks)

    # To [1, 3, n]
    xs = [l[0] for l in landmarks]
    ys = [l[1] for l in landmarks]
    zs = [l[2] for l in landmarks]
    tensor = torch.tensor([xs, ys, zs], dtype=torch.float32).unsqueeze(0)  # [1, 3, N]

    if current_mode == "train":
        data_buffer.append(tensor)
        label_buffer.append(int(class_id))
        print(f"> Stored training sample for class {class_id}")

    elif current_mode == "inference":
        model.eval()
        with torch.no_grad():
            logits, feat_embed = model(tensor)          # logits: [1, C]
            probs = torch.softmax(logits, dim=-1)       # [1, C]
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()

            # Always send confidence; only send class if >= threshold
            client.send_message("/confidence", conf)
            if conf >= CONFIDENCE_THRESHOLD:
                client.send_message("/predicted_class", pred)

                # Verbose state transitions
                if infer_verbose:
                    if current_active_class is None:
                        print(f"> Gesture START: class {pred} (conf {conf:.2f})")
                        current_active_class = pred
                    elif current_active_class != pred:
                        print(f"> Gesture CHANGE: {current_active_class} -> {pred} (conf {conf:.2f})")
                        current_active_class = pred
            else:
                # Below threshold: suppress class, but log an END once
                if infer_verbose and current_active_class is not None:
                    print(f"> Gesture END: class {current_active_class} (conf {conf:.2f} < {CONFIDENCE_THRESHOLD})")
                    current_active_class = None

            # Compact features (length = EMBED_DIM)
            client.send_message("/features", feat_embed[0].tolist())

def train_model(addr, *args):
    global data_buffer, label_buffer
    if not data_buffer:
        print("> No data to train.")
        return

    # Stack dataset
    X = torch.cat(data_buffer, dim=0)                 # [N, 3, L]
    y = torch.tensor(label_buffer, dtype=torch.long)  # [N]
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"> Dataset size: {len(dataset)}, Batch size: {batch_size}, Batches/epoch: {len(loader)}")

    for epoch in range(training_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            if train_verbose:
                print(f"  Epoch {epoch+1}  Batch {batch_idx}/{len(loader)}  Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataset)
        print(f"> Epoch {epoch+1}/{training_epochs} - Loss: {epoch_loss:.4f}")

    print("> Training complete.")

def save_model(addr, *args):
    if model is not None:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"> Model saved to {MODEL_SAVE_PATH}")

def load_model(addr, *args):
    if model is None:
        print("> Model not initialized.")
        return
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"> No saved model at {MODEL_SAVE_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()
    print(f"> Model loaded from {MODEL_SAVE_PATH}")

# =========================
# OSC WIRING
# =========================
def setup_osc():
    disp = dispatcher.Dispatcher()
    disp.map("/set_mode", set_mode)
    disp.map("/set_classes", set_class_count)
    disp.map("/normalize", set_normalization)
    disp.map("/normalize_scale", set_normalize_scale)
    disp.map("/normalize_rotate", set_normalize_rotate)
    disp.map("/set_epochs", set_epochs)
    disp.map("/set_batch_size", set_batch_size)
    disp.map("/set_confidence_threshold", set_confidence_threshold)
    disp.map("/set_train_verbose", set_train_verbose)
    disp.map("/set_infer_verbose", set_infer_verbose)

    disp.map("/reset_model", reset_model)
    disp.map("/clear_training", clear_training)

    disp.map("/input", receive_sample)   # canonical address for both train/infer samples
    disp.map("/train", train_model)
    disp.map("/save_model", save_model)
    disp.map("/load_model", load_model)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", OSC_RECEIVE_PORT), disp)
    print(f"> Listening on port {OSC_RECEIVE_PORT}")
    server.serve_forever()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    init_model(landmark_len, class_count)
    setup_osc()
