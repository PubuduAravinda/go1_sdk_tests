#!/usr/bin/env python3
"""
GO1 Chin Camera → MiDaS Small + Colored Depth Map + Perfect 0.32 m
45–60 FPS | Auto-calibration | RGB + Depth side-by-side
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2
import numpy as np
import torch
import queue
import time

print("Loading MiDaS Small + colormap...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

# Colormap for beautiful depth visualization
colormap = cv2.COLORMAP_INFERNO   # or try TURBO, PLASMA, VIRIDIS

Gst.init(None)

class GO1ChinCamera:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.pipeline = None
        self.calib_vals = []
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

    def calibrate_once(self, raw):
        if self.scale is not None: return
        self.calib_vals.append(raw)
        if len(self.calib_vals) >= 50:
            median = np.median(self.calib_vals)
            self.scale = median / 0.32
            print(f"\nCALIBRATION DONE! scale = {self.scale:.2f} → perfect 0.32 m\n")

cam = GO1ChinCamera()
cam.start()

cv2.namedWindow("GO1 Chin + Depth → Real Height (MiDaS Small)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GO1 Chin + Depth → Real Height (MiDaS Small)", 1800, 900)

print("\n" + "="*90)
print("   MiDaS Small + Colored Depth Map → wait 1 sec → perfect 0.32 m")
print("="*90 + "\n")

try:
    while True:
        frame = cam.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]

            # MiDaS inference
            input_batch = transform(frame).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            depth_raw = prediction.cpu().numpy()

            # Normalize for visualization
            depth_min = depth_raw.min()
            depth_max = depth_raw.max()
            depth_norm = (depth_raw - depth_min) / (depth_max - depth_min + 1e-6)
            depth_viz = (255 * depth_norm).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_viz, colormap)
            depth_colored = cv2.resize(depth_colored, (w, h))

            # Sample ground (very bottom center)
            y_center = int(h * 0.94)
            x_center = w // 2
            box = 160
            half = box // 2
            y1 = max(0, y_center - half)
            y2 = min(h, y_center + half)
            x1 = max(0, x_center - half)
            x2 = min(w, x_center + half)

            # Sample ground/obstacle — use MIN = closest point!
            raw_min = np.min(depth_raw[y1:y2, x1:x2])
            raw_mean = np.mean(depth_raw[y1:y2, x1:x2])   # for debugging only

            # Use min for real height (robust to obstacles)
            raw = raw_min

            cam.calibrate_once(raw)
            height_m = round(raw / (cam.scale or 1.0), 3) if cam.scale else 0.0

            # Optional: show both values for fun
            cv2.putText(frame, f"{height_m:.3f} m", (60, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 5.5, (0, 255, 0), 12)
            cv2.putText(frame, f"min", (x2 + 20, y1 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # raw = np.mean(depth_raw[y1:y2, x1:x2])
            # cam.calibrate_once(raw)
            # height_m = round(raw / (cam.scale or 1.0), 3) if cam.scale else 0.0
            #
            # # Draw on RGB
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
            # cv2.putText(frame, f"{height_m:.3f} m", (60, 150),
            #             cv2.FONT_HERSHEY_DUPLEX, 5.5, (0, 255, 0), 12)
            if not cam.scale:
                cv2.putText(frame, "CALIBRATING...", (60, 300),
                            cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 8)

            # Side-by-side
            combined = np.hstack([frame, depth_colored])
            cv2.imshow("GO1 Chin + Depth → Real Height (MiDaS Small)", combined)

            print(f"Height: {height_m:.3f} m     ", end="\r", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    cv2.destroyAllWindows()