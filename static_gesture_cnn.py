import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pythonosc import dispatcher, osc_server, udp_client

# ========== CONFIG ========== #
OSC_RECEIVE_PORT = 9000
OSC_SEND_PORT = 9001
OSC_SEND_IP = "127.0.0.1"
MODEL_SAVE_PATH = "gesture_model.pt"

# ========== PARAMETERS ========== #
landmark_len = 21          # Default number of landmarks
class_count = 3            # Default number of classes
training_epochs = 100      # Default number of training epochs
normalize_origin = True    # Normalize to first landmark
CONFIDENCE_THRESHOLD = 0.7 # Threshold for prediction output

# ========== STATE ========== #
data_buffer = []
label_buffer = []
current_mode = "inference"
model = None
optimizer = None
loss_fn = nn.CrossEntropyLoss()
client = udp_client.SimpleUDPClient(OSC_SEND_IP, OSC_SEND_PORT)

# ========== MODEL ========== #
class GestureCNN(nn.Module):
    def __init__(self, input_len: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        features = x.clone()
        x = self.pool(x)
        x = x.squeeze(2)
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs, features

def init_model(n_landmarks: int, n_classes: int):
    global model, optimizer, landmark_len, class_count
    landmark_len = n_landmarks
    class_count = n_classes
    model = GestureCNN(landmark_len, class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"> Model initialized with {n_landmarks} landmarks and {n_classes} classes.")

# ========== OSC HANDLERS ========== #
def set_mode(addr, *args):
    global current_mode
    if len(args) < 1:
        print("> Error: /set_mode requires 'train' or 'inference'")
        return
    mode = args[0]
    if mode in ("train", "inference"):
        current_mode = mode
        print(f"> Mode set to: {mode}")
    else:
        print(f"> Invalid mode: {mode}")

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

def set_epochs(addr, *args):
    global training_epochs
    if len(args) < 1:
        print("> Error: /set_epochs requires an integer")
        return
    training_epochs = int(args[0])
    print(f"> Training epochs set to: {training_epochs}")

def set_confidence_threshold(addr, *args):
    global CONFIDENCE_THRESHOLD
    if len(args) < 1:
        print("> Error: /set_confidence_threshold requires a float")
        return
    CONFIDENCE_THRESHOLD = float(args[0])
    print(f"> Confidence threshold set to: {CONFIDENCE_THRESHOLD:.2f}")

def reset_model(addr, *args):
    init_model(landmark_len, class_count)
    print("> Model reset to initial weights.")

def clear_training(addr, *args):
    global data_buffer, label_buffer
    data_buffer = []
    label_buffer = []
    print("> Cleared all stored training data.")

def receive_sample(addr, *args):
    global model
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

    if model is None or num_landmarks != landmark_len:
        init_model(num_landmarks, class_count)

    landmarks = list(zip(x_vals, y_vals, z_vals))

    if normalize_origin:
        origin = landmarks[0]
        landmarks = [[x - origin[0], y - origin[1], z - origin[2]] for x, y, z in landmarks]

    x = [l[0] for l in landmarks]
    y = [l[1] for l in landmarks]
    z = [l[2] for l in landmarks]
    tensor = torch.tensor([x, y, z], dtype=torch.float32).unsqueeze(0)  # [1, 3, n]

    print(f"> Received sample (class {class_id}) in '{current_mode}' mode with {num_landmarks} landmarks.")

    if current_mode == "train":
        data_buffer.append(tensor)
        label_buffer.append(torch.tensor(class_id))
        print(f"> Stored training sample for class {class_id}")

    elif current_mode == "inference":
        with torch.no_grad():
            output, features = model(tensor)
            pred = torch.argmax(output, dim=-1).item()
            conf = torch.max(output).item()
            client.send_message("/confidence", conf)

            if conf >= CONFIDENCE_THRESHOLD:
                client.send_message("/predicted_class", pred)
            else:
                print(f"> Confidence {conf:.2f} below threshold ({CONFIDENCE_THRESHOLD}), suppressing prediction.")

            client.send_message("/features", features.flatten().tolist())

def train_model(addr, *args):
    global data_buffer, label_buffer
    if not data_buffer:
        print("> No data to train.")
        return

    X = torch.cat(data_buffer, dim=0)
    y = torch.tensor(label_buffer)

    for epoch in range(training_epochs):
        optimizer.zero_grad()
        out, _ = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        print(f"> Epoch {epoch+1}/{training_epochs} - Loss: {loss.item():.4f}")

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
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print(f"> Model loaded from {MODEL_SAVE_PATH}")

# ========== OSC SETUP ========== #
def setup_osc():
    disp = dispatcher.Dispatcher()
    disp.map("/set_mode", set_mode)
    disp.map("/set_classes", set_class_count)
    disp.map("/normalize", set_normalization)
    disp.map("/set_epochs", set_epochs)
    disp.map("/set_confidence_threshold", set_confidence_threshold)
    disp.map("/reset_model", reset_model)
    disp.map("/clear_training", clear_training)
    disp.map("/input", receive_sample)
    disp.map("/train", train_model)
    disp.map("/save_model", save_model)
    disp.map("/load_model", load_model)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", OSC_RECEIVE_PORT), disp)
    print(f"> Listening on port {OSC_RECEIVE_PORT}")
    server.serve_forever()

# ========== MAIN ========== #
if __name__ == "__main__":
    init_model(landmark_len, class_count)
    setup_osc()
