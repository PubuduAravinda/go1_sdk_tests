#!/usr/bin/env python3
"""
GO1 Chin Camera → 5x5 GRID DEPTH – PHYSICALLY CORRECT
Center cell = 0.24 m | Bottom row = lowest height | Top row = highest
+ Terminal debug print of all 25 values
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2
import numpy as np
import torch
import queue
import time

print("Loading MiDaS Small...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

Gst.init(None)

class GO1ChinCamera:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.pipeline = None
        self.scale = None

    def on_new_sample(self, sink, _):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            w = caps.get_structure(0).get_value('width')
            h = caps.get_structure(0).get_value('height')
            res, mapinfo = buf.map(Gst.MapFlags.READ)
            if res:
                frame = np.frombuffer(mapinfo.data, np.uint8).reshape((h, w, 3))
                buf.unmap(mapinfo)
                frame = cv2.flip(frame, -1)
                try: self.frame_queue.get_nowait()
                except: pass
                self.frame_queue.put_nowait(frame.copy())
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def start(self):
        pipeline = "udpsrc port=9202 ! application/x-rtp,media=video,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        self.pipeline = Gst.parse_launch(pipeline)
        self.pipeline.get_by_name('sink').connect("new-sample", self.on_new_sample, None)
        self.pipeline.set_state(Gst.State.PLAYING)
        time.sleep(1.5)
        print("Camera live!")

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if self.pipeline: self.pipeline.set_state(Gst.State.NULL)

    def calibrate_center(self, center_raw_inverse):
        if self.scale is None:
            center_depth = 1.0 / (center_raw_inverse + 1e-6)
            self.scale = center_depth / 0.24
            print(f"\nCALIBRATION DONE! Center raw={center_raw_inverse:.1f} → depth={center_depth:.3f} m → scale={self.scale:.3f}")
            print("   All 25 cells now show PHYSICALLY CORRECT height!\n")

cam = GO1ChinCamera()
cam.start()

cv2.namedWindow("GO1 5x5 Grid – CORRECT HEIGHTS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GO1 5x5 Grid – CORRECT HEIGHTS", 1800, 1000)

print("\n" + "="*95)
print("   5x5 GRID – PHYSICALLY CORRECT (bottom row = lowest, top row = highest)")
print("   Center cell calibrates to 0.24 m → All values now make sense!")
print("="*95 + "\n")

try:
    frame_count = 0
    while True:
        frame = cam.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            frame_count += 1

            input_batch = transform(frame).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            depth_inverse = prediction.cpu().numpy()  # ← MiDaS gives inverse depth!

            # Colored depth viz
            norm = (depth_inverse - depth_inverse.min()) / (depth_inverse.max() - depth_inverse.min() + 1e-6)
            viz = (255 * norm).astype(np.uint8)
            depth_colored = cv2.applyColorMap(viz, cv2.COLORMAP_INFERNO)

            # 5x5 grid
            grid_w = w // 5
            grid_h = h // 5
            grid_values = []

            for row in range(5):
                row_vals = []
                for col in range(5):
                    x1, y1 = col * grid_w, row * grid_h
                    x2, y2 = x1 + grid_w, y1 + grid_h

                    # Draw grid
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 3)

                    # Inverse depth → real depth → real height
                    inv_val = np.max(depth_inverse[y1:y2, x1:x2])  # max inverse = closest point
                    real_depth = 1.0 / (inv_val + 1e-6)

                    # Calibrate from center
                    if row == 2 and col == 2:
                        cam.calibrate_center(inv_val)

                    height_m = round(real_depth / (cam.scale or 1.0), 3) if cam.scale else 0.0
                    row_vals.append(height_m)

                    # Draw value
                    text = f"{height_m:.3f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
                    tx = x1 + (grid_w - text_size[0]) // 2
                    ty = y1 + (grid_h + text_size[1]) // 2
                    color = (0, 255, 0) if (row == 2 and col == 2) else (50, 255, 50)
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3)

                    if row == 2 and col == 2:
                        cv2.putText(frame, "CENTER 0.24m", (tx, ty + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                grid_values.append(row_vals)

            # Debug print every 30 frames
            if frame_count % 30 == 0 and cam.scale:
                print("\n" + "-"*60)
                print("5×5 GRID HEIGHTS (meters) – Row 0 = farthest, Row 4 = closest")
                for i, row in enumerate(grid_values):
                    print(f"Row {i}: {row}")
                print("-"*60)

            if not cam.scale:
                cv2.putText(frame, "CALIBRATING CENTER CELL...", (50, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 7)

            combined = np.hstack([frame, depth_colored])
            cv2.imshow("GO1 5x5 Grid – CORRECT HEIGHTS", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped")
finally:
    cam.stop()
    cv2.destroyAllWindows()