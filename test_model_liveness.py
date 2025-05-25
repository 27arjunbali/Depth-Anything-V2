import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch
torch.backends.mps.is_available = lambda : False
torch.backends.mps.is_built = lambda : False
import numpy as np
import pandas as pd
from depth_anything_v2.dpt import DepthAnythingV2
from skimage.feature import local_binary_pattern
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def extract_features(face_img, model, input_size):
    face_img = face_img.copy()  # Ensure no shared memory with OpenCV
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness_std = np.std(face_gray) / 255.0
    lbp = local_binary_pattern(face_gray, P=8, R=1, method='uniform')
    lbp_std = np.std(lbp)
    edges = cv2.Canny(face_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    depth = model.infer_image(face_img, input_size)
    depth_resized = cv2.resize(depth, (face_img.shape[1], face_img.shape[0]))
    norm_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
    depth_std = np.std(norm_depth)
    depth_range = np.max(norm_depth) - np.min(norm_depth)
    depth_mean = np.mean(norm_depth)

    # Landmark-based 3D distances
    results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    h, w = face_img.shape[:2]
    dist_left, dist_right = np.nan, np.nan
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        def xy(landmark): return int(landmark.x * w), int(landmark.y * h)
        try:
            nx, ny = xy(lm[1])
            lex, ley = xy(lm[234])
            rex, rey = xy(lm[454])
            depth_left = norm_depth[ley, lex]
            depth_right = norm_depth[rey, rex]
            depth_nose = norm_depth[ny, nx]
            dist_left = np.linalg.norm([nx - lex, ny - ley, depth_nose - depth_left])
            dist_right = np.linalg.norm([nx - rex, ny - rey, depth_nose - depth_right])
        except:
            pass
    features_array = np.array([[brightness_std, lbp_std, edge_density,
                               depth_std, depth_range, depth_mean,
                               dist_left, dist_right, 0.0]], dtype=np.float32)
    return torch.from_numpy(features_array).to(torch.float32).cpu()

def main():
    # Load MLP model
    csv = pd.read_csv("features_dataset.csv")
    input_dim = csv.shape[1] - 1
    model = SimpleMLP(input_dim)
    model.load_state_dict(torch.load("transformer_model.pt", map_location='cpu'))
    model.eval()
    model.to('cpu')

    # Load depth model
    depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vits.pth", map_location='cpu'))
    depth_model.eval()

    cap = cv2.VideoCapture(1)

    print("[INFO] Running liveness detector... Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        label_text = "No face detected"
        box_color = (0, 0, 255)  # Red by default

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            lm = results.multi_face_landmarks[0].landmark
            xs = [int(p.x * w) for p in lm]
            ys = [int(p.y * h) for p in lm]
            x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
            x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            label_text = "Face detected"
            box_color = (0, 255, 0)

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and results.multi_face_landmarks:
            face_img = frame[y1:y2, x1:x2]
            try:
                features = extract_features(face_img, depth_model, 256)
                features_tensor = features.float().cpu()
                with torch.no_grad():
                    logits = model(features_tensor)
                    pred = logits.argmax(dim=1).item()
                    label_text = "LIVE" if pred == 1 else "SPOOF"
                    color = (0, 255, 0) if pred == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    print(f"[DEBUG] Face captured and prediction made: {label_text}")
            except Exception as e:
                print("[WARN] Feature extraction failed:", e)

        cv2.imshow("Liveness Test", frame)

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()