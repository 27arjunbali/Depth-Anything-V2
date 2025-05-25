from sklearn.metrics import f1_score
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def train_model():
    # Load dataset
    csv_file = "features_dataset.csv"
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['brightness_std', 'lbp_std', 'edge_density', 'depth_std', 'depth_range', 'depth_mean',
                  'dist_nose_left', 'dist_nose_right', 'temporal_depth_variance', 'label']
    # Remove rows with NaN
    df = df.dropna()
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    class FeaturesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Augment dataset by sampling with replacement up to 10,000 samples
    n_samples = len(X)
    augmented_size = max(n_samples, 10000)
    # Balance classes 50% real (1) and 50% spoof (0)
    real_indices = np.where(y == 1)[0]
    spoof_indices = np.where(y == 0)[0]
    half_size = augmented_size // 2
    real_sampled = np.random.choice(real_indices, size=half_size, replace=True) if len(real_indices) > 0 else np.array([], dtype=int)
    spoof_sampled = np.random.choice(spoof_indices, size=augmented_size - half_size, replace=True) if len(spoof_indices) > 0 else np.array([], dtype=int)
    indices = np.concatenate([real_sampled, spoof_sampled])
    np.random.shuffle(indices)
    X_aug = X[indices]
    y_aug = y[indices]
    print(f"[DEBUG] Augmented dataset real: {np.sum(y_aug == 1)}, spoof: {np.sum(y_aug == 0)}")

    dataset = FeaturesDataset(X_aug, y_aug)

    # Split into 80% train, 20% eval
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.model(x)

    input_dim = X.shape[1]
    num_classes = 2
    model = SimpleMLP(input_dim, num_classes)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    print("[INFO] Starting training on collected features...")
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            preds = out.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
        # Step the scheduler after the epoch
        scheduler.step()
        acc = correct / total if total > 0 else 0

        # Evaluate on eval set per epoch
        model.eval()
        eval_total = 0
        eval_correct = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                eval_total += yb.size(0)
                eval_correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
        # Debug: print prediction distribution
        unique_preds = np.unique(all_preds, return_counts=True)
        print(f"[DEBUG] Preds distribution: {dict(zip(*unique_preds))}")
        eval_acc = eval_correct / eval_total if eval_total > 0 else 0
        eval_f1 = f1_score(all_targets, all_preds, average="binary")
        print(f"Epoch {epoch+1}/{epochs} - Training accuracy: {acc:.4f} | Eval accuracy: {eval_acc:.4f} | Eval F1 Score: {eval_f1:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "transformer_model.pt")
    print("[INFO] Trained model saved as transformer_model.pt")

import cv2
import numpy as np
import os
import torch
import argparse
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
from skimage.feature import local_binary_pattern
import mediapipe as mp
import csv

def normalize_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--depth-threshold', type=float, default=5.0)
    parser.add_argument('--depth-threshold-std', type=float, default=0.03)
    parser.add_argument('--depth-threshold-range', type=float, default=0.05)
    parser.add_argument('--depth-threshold-mean-min', type=float, default=0.2)
    parser.add_argument('--depth-threshold-mean-max', type=float, default=0.8)
    args = parser.parse_args()

    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    print("[INFO] Loading model...")
    model = DepthAnythingV2(**model_configs[args.encoder])
    model_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(DEVICE).eval()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    print("[INFO] Starting webcam... Press 'c' to capture face for labeling, 'q' to quit, 't' to train model.")
    print("After capture, press 'y' for real face, 'n' for spoof/photo to label and save features.")

    captured_image = None
    captured_face = None
    captured_features = None
    capture_mode = False

    depth_history = []
    fixed_depth_size = None
    temporal_window = 5  # number of frames to consider
    temporal_var_threshold = 0.001  # adjustable threshold for temporal variance

    csv_file = "features_dataset.csv"
    csv_header = ['brightness_std', 'lbp_std', 'edge_density', 'depth_std', 'depth_range', 'depth_mean',
                  'dist_nose_left', 'dist_nose_right', 'temporal_depth_variance', 'label']

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    break_loop = False
    train_requested = False
    while True:
        if not capture_mode:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, "Press 'c' to capture face for labeling", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Depth + Liveness Detection - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if len(faces) > 0:
                    with torch.no_grad():
                        x, y, w, h = faces[0]
                        face_img = frame[y:y+h, x:x+w]
                        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                        brightness_std = np.std(face_gray) / 255.0
                        lbp = local_binary_pattern(face_gray, P=8, R=1, method='uniform')
                        lbp_std = np.std(lbp)
                        edges = cv2.Canny(face_gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size

                        depth = model.infer_image(face_img, args.input_size)
                        if depth is not None:
                            if depth.shape != (h, w):
                                depth_resized = cv2.resize(depth, (w, h))
                            else:
                                depth_resized = depth

                            min_depth = 0.1
                            max_depth = 10.0
                            clipped_depth = np.clip(depth_resized, min_depth, max_depth)
                            norm_depth = (clipped_depth - clipped_depth.min()) / (clipped_depth.max() - clipped_depth.min() + 1e-8)

                            depth_std = np.std(norm_depth)
                            depth_range = np.max(norm_depth) - np.min(norm_depth)
                            depth_mean = np.mean(norm_depth)

                            normalized_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
                            if fixed_depth_size is None:
                                fixed_depth_size = (w, h)

                            resized_depth = cv2.resize(normalized_depth, fixed_depth_size)
                            depth_history.append(resized_depth)
                            if len(depth_history) > temporal_window:
                                depth_history.pop(0)

                            temporal_depth_variance = None
                            if len(depth_history) == temporal_window:
                                temporal_var_map = np.var(np.stack(depth_history), axis=0)
                                temporal_depth_variance = np.mean(temporal_var_map)
                            else:
                                temporal_depth_variance = np.nan
                        else:
                            depth_std = np.nan
                            depth_range = np.nan
                            depth_mean = np.nan
                            temporal_depth_variance = np.nan
                            resized_depth = None

                        # MediaPipe Face Mesh landmark distance check
                        threshold_3d_dist = 15.0

                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(face_img_rgb)

                        dist_nose_left = np.nan
                        dist_nose_right = np.nan

                        if results.multi_face_landmarks and resized_depth is not None:
                            face_landmarks = results.multi_face_landmarks[0]
                            h_f, w_f, _ = face_img.shape

                            def get_landmark_coords(landmark):
                                return int(landmark.x * w_f), int(landmark.y * h_f)

                            lm_nose = face_landmarks.landmark[1]
                            lm_left_ear = face_landmarks.landmark[234]
                            lm_right_ear = face_landmarks.landmark[454]

                            nx, ny = get_landmark_coords(lm_nose)
                            lex, ley = get_landmark_coords(lm_left_ear)
                            rex, rey = get_landmark_coords(lm_right_ear)

                            dh, dw = resized_depth.shape
                            nx = np.clip(nx, 0, dw - 1)
                            ny = np.clip(ny, 0, dh - 1)
                            lex = np.clip(lex, 0, dw - 1)
                            ley = np.clip(ley, 0, dh - 1)
                            rex = np.clip(rex, 0, dw - 1)
                            rey = np.clip(rey, 0, dh - 1)

                            depth_nose = resized_depth[ny, nx]
                            depth_left_ear = resized_depth[ley, lex]
                            depth_right_ear = resized_depth[rey, rex]

                            dist_nose_left = np.sqrt((nx - lex)**2 + (ny - ley)**2 + (depth_nose - depth_left_ear)**2)
                            dist_nose_right = np.sqrt((nx - rex)**2 + (ny - rey)**2 + (depth_nose - depth_right_ear)**2)

                        captured_image = frame.copy()
                        captured_face = face_img.copy()
                        captured_features = [brightness_std, lbp_std, edge_density, depth_std, depth_range, depth_mean,
                                             dist_nose_left, dist_nose_right, temporal_depth_variance]
                        # Validate that all features are not NaN
                        if any(np.isnan(v) for v in captured_features):
                            print("[WARN] One or more features could not be computed. Try again.")
                            capture_mode = False
                            continue
                        capture_mode = True
                else:
                    print("[INFO] No face detected. Try again.")

            elif key == ord('q'):
                break_loop = True
                break
            elif key == ord('t'):
                break_loop = True
                train_requested = True
                break

        else:
            display_img = captured_face.copy()
            cv2.putText(display_img, "Press 'y' = real face, 'n' = spoof/photo, 'c' = cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Captured Face - Label and Save", display_img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y') or key == ord('n'):
                label = 1 if key == ord('y') else 0
                if captured_features is not None:
                    row = captured_features + [label]
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"[INFO] Features and label saved: {row}")
                else:
                    print("[WARN] No features extracted to save.")
                capture_mode = False
                captured_features = None
                captured_face = None
                captured_image = None

            elif key == ord('c'):
                capture_mode = False
                captured_features = None
                captured_face = None
                captured_image = None
                print("[INFO] Capture cancelled. Returning to live feed.")

            elif key == ord('q'):
                break_loop = True
                break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    if train_requested:
        train_model()

if __name__ == '__main__':
    main()