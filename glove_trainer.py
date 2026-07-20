
import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys
from collections import deque
from model import FlexGloveCNN


SERIAL_PORT   = "COM3"        
BAUD_RATE     = 115200
NUM_SENSORS   = 5
WINDOW_SIZE   = 30            
HOLD_SECONDS  = 3            
SAMPLE_HZ     = 50            
NUM_WORDS     = 50          
SAVE_PATH     = "glove_model.pt"
LR            = 1e-3


def load_word_list(n: int) -> list[str]:
   
    try:
        import nltk
        from nltk.corpus import words as nltk_words
        nltk.download("words", quiet=True)
        all_words = nltk_words.words()
       
        filtered = [w.lower() for w in all_words
                    if 3 <= len(w) <= 6 and w.isalpha() and w.islower()]
        random.seed(42)
        return random.sample(filtered, min(n, len(filtered)))
    except ImportError:
        print("nltk not found. pip install nltk")
        sys.exit(1)


def find_serial_port() -> str:
   
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description or "usbserial" in p.device:
            return p.device
    return SERIAL_PORT


def read_frame(ser: serial.Serial) -> np.ndarray | None:
   
    try:
        line = ser.readline().decode("utf-8").strip()
        vals = [float(v) for v in line.split(",")]
        if len(vals) == NUM_SENSORS:
            return np.array(vals, dtype=np.float32)
    except Exception:
        pass
    return None


def collect_gesture(ser: serial.Serial, duration_s: float = HOLD_SECONDS) -> np.ndarray:
   
    ser.reset_input_buffer()
    frames = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        frame = read_frame(ser)
        if frame is not None:
            frames.append(frame)
    # Pad or truncate to WINDOW_SIZE
    if len(frames) >= WINDOW_SIZE:
        # Take middle window
        start = (len(frames) - WINDOW_SIZE) // 2
        frames = frames[start:start + WINDOW_SIZE]
    else:
        # Pad with last frame
        while len(frames) < WINDOW_SIZE:
            frames.append(frames[-1] if frames else np.zeros(NUM_SENSORS))
    return np.array(frames, dtype=np.float32)  # (WINDOW_SIZE, NUM_SENSORS)


def normalize(window: np.ndarray) -> np.ndarray:
   
    mn = window.min(axis=0, keepdims=True)
    mx = window.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
    return (window - mn) / rng


def train_step(
    model: FlexGloveCNN,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    window: np.ndarray,
    label_idx: int,
    device: torch.device,
) -> tuple[float, int]:
   
    model.train()
    x = torch.tensor(normalize(window)).unsqueeze(0).to(device)   # (1, W, S)
    y = torch.tensor([label_idx], dtype=torch.long).to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    pred = logits.argmax(dim=1).item()
    return loss.item(), pred


def predict(
    model: FlexGloveCNN,
    window: np.ndarray,
    device: torch.device,
) -> int:
  
    model.eval()
    with torch.no_grad():
        x = torch.tensor(normalize(window)).unsqueeze(0).to(device)
        logits = model(x)
        return logits.argmax(dim=1).item()


def print_banner(text: str, char: str = "─") -> None:
    w = min(60, os.get_terminal_size().columns)
    print(char * w)
    print(text.center(w))
    print(char * w)


def main():
    print_banner("FLEX GLOVE TRAINER", "═")

  
    print("Loading word list...")
    word_list = load_word_list(NUM_WORDS)
    num_classes = len(word_list)
    print(f"Loaded {num_classes} words: {word_list[:10]}...")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = FlexGloveCNN(NUM_SENSORS, WINDOW_SIZE, num_classes).to(device)
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        print(f"Loaded existing model from {SAVE_PATH}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    
    port = find_serial_port()
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)  # wait for Arduino reset
    print("Connected.\n")

 
    session_correct = 0
    session_total   = 0

    try:
        while True:
           
            label_idx   = random.randint(0, num_classes - 1)
            target_word = word_list[label_idx]

            print_banner(f'Sign the word:  "{target_word.upper()}"')
            print(f"You have {HOLD_SECONDS} seconds. Starting in 2s...")
            time.sleep(2)
            print(">>> RECORDING <<<")

            window = collect_gesture(ser, HOLD_SECONDS)

            print("Processing...")
            loss, pred_idx = train_step(model, optimizer, criterion, window, label_idx, device)
            pred_word      = word_list[pred_idx]
            correct        = pred_idx == label_idx

            session_total   += 1
            session_correct += int(correct)
            accuracy         = session_correct / session_total * 100

            print(f"\n  Target  : {target_word.upper()}")
            print(f"  Model   : {pred_word.upper()}  {'✓ CORRECT' if correct else '✗ wrong'}")
            print(f"  Loss    : {loss:.4f}")
            print(f"  Accuracy: {accuracy:.1f}%  ({session_correct}/{session_total})")

            
            torch.save(model.state_dict(), SAVE_PATH)

            print("\nPress Enter for next word, or Ctrl+C to quit...")
            input()

    except KeyboardInterrupt:
        print("\nSession ended.")
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")
        ser.close()


if __name__ == "__main__":
    main()
