import cv2
import numpy as np
import os
import torch
import argparse
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
from skimage.feature import local_binary_pattern

def normalize_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', action='store_true')
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

    print("[INFO] Starting webcam... Press 'c' to capture, 'q' to quit.")

    live_label = "NOT LIVE"
    live_color = (0, 0, 255)

    captured_image = None
    capture_mode = False

    while True:
        if not capture_mode:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, f"Liveness: {live_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, live_color, 3)
            cv2.imshow("Depth + Liveness Detection", frame)

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
                        print(f"[DEBUG] Brightness std deviation: {brightness_std:.4f}")
                        print(f"[DEBUG] LBP std deviation: {lbp_std:.2f}")
                        edges = cv2.Canny(face_gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        print(f"[DEBUG] Edge density: {edge_density:.4f}")
                        if brightness_std >= 0.20 and lbp_std > 2.4 and edge_density > 0.02:
                            depth = model.infer_image(face_img, args.input_size)
                            if depth is not None:
                                # Resize depth map to face region size if needed
                                if depth.shape != (h, w):
                                    depth_resized = cv2.resize(depth, (w, h))
                                else:
                                    depth_resized = depth
                                import matplotlib.pyplot as plt
                                plt.imshow(depth_resized, cmap='plasma')
                                plt.title("Depth Map")
                                plt.show()

                                min_depth = 0.1
                                max_depth = 10.0
                                clipped_depth = np.clip(depth_resized, min_depth, max_depth)
                                norm_depth = (clipped_depth - clipped_depth.min()) / (clipped_depth.max() - clipped_depth.min() + 1e-8)

                                std_dev = np.std(norm_depth)
                                depth_range = np.max(norm_depth) - np.min(norm_depth)
                                depth_mean = np.mean(norm_depth)

                                print(f"[DEBUG] std: {std_dev:.4f}, range: {depth_range:.4f}, mean: {depth_mean:.4f}")

                                if (std_dev > args.depth_threshold_std and
                                    depth_range > args.depth_threshold_range and
                                    args.depth_threshold_mean_min < depth_mean < args.depth_threshold_mean_max):
                                    live_label = "LIVE"
                                    live_color = (0, 255, 0)
                                else:
                                    live_label = "NOT LIVE"
                                    live_color = (0, 0, 255)
                            else:
                                live_label = "NOT LIVE"
                                live_color = (0, 0, 255)
                        else:
                            live_label = "NOT LIVE"
                            live_color = (0, 0, 255)
                else:
                    live_label = "NOT LIVE"
                    live_color = (0, 0, 255)

                captured_image = frame.copy()
                capture_mode = True

            elif key == ord('q'):
                break

        else:
            display_img = captured_image.copy()
            cv2.putText(display_img, f"Liveness: {live_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, live_color, 3)
            cv2.imshow("Depth + Liveness Detection", display_img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):
                save_dir = "captured_images"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                filename = f"{live_label}_{int(cv2.getTickCount())}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, display_img)
                print(f"[INFO] Image saved to {save_path}")

            elif key == ord('c'):
                capture_mode = False
                live_label = "NOT LIVE"
                live_color = (0, 0, 255)

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()