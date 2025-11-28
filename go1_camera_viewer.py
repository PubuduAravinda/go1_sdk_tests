#!/usr/bin/env python3
"""
GO1 Robot Camera Viewer for Laptop (192.168.123.100)

This script receives and displays camera streams from the GO1 Head Nano board.

Usage:
    python3 go1_camera_viewer.py --camera face    # View face camera (port 9201)
    python3 go1_camera_viewer.py --camera chin    # View chin camera (port 9202)
    python3 go1_camera_viewer.py --camera both    # View both cameras side-by-side

Requirements:
    - GStreamer with H264 support
    - Python packages: gi, numpy, opencv-python

Install:
    sudo apt-get install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                            gstreamer1.0-plugins-ugly gstreamer1.0-libav \
                            python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0
    pip3 install numpy opencv-python
"""

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import queue
import time
import argparse
import sys

Gst.init(None)


class CameraStream:
    """Handles a single camera stream from GO1"""

    def __init__(self, port, name):
        self.port = port
        self.name = name
        self.frame_queue = queue.Queue(maxsize=1)
        self.pipeline = None
        self.is_running = False

    def on_new_sample(self, sink, data):
        """Callback when new frame arrives"""
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')

            result, mapinfo = buf.map(Gst.MapFlags.READ)
            if result:
                frame_data = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
                buf.unmap(mapinfo)

                # Keep only the latest frame
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

                self.frame_queue.put_nowait(frame_data)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def start(self):
        """Start the camera stream"""
        pipeline_desc = (
            f'udpsrc port={self.port} '
            'caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! '
            'appsink name=sink emit-signals=true max-buffers=1 drop=true'
        )

        print(f"Starting {self.name} stream on port {self.port}...")
        self.pipeline = Gst.parse_launch(pipeline_desc)
        appsink = self.pipeline.get_by_name('sink')
        appsink.connect("new-sample", self.on_new_sample, None)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True
        print(f"✓ {self.name} stream started")

    def get_frame(self):
        """Get the latest frame"""
        try:
            frame = self.frame_queue.get_nowait()
            # Flip frame (GO1 cameras are mounted upside down)
            return cv2.flip(frame, -1)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the camera stream"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            print(f"✓ {self.name} stream stopped")


def view_single_camera(camera_type):
    """View a single camera stream"""
    if camera_type == 'face':
        stream = CameraStream(9201, "Face Camera")
    else:  # chin
        stream = CameraStream(9202, "Chin Camera")

    stream.start()

    print(f"\nViewing {stream.name}...")
    print("Press 'q' to quit, 's' to save snapshot")

    snapshot_count = 0
    last_frame = None

    try:
        while True:
            frame = stream.get_frame()

            if frame is not None:
                last_frame = frame
                cv2.imshow(f"GO1 - {stream.name}", frame)
            elif last_frame is not None:
                cv2.imshow(f"GO1 - {stream.name}", last_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_frame is not None:
                filename = f"{camera_type}_snapshot_{snapshot_count}.jpg"
                cv2.imwrite(filename, last_frame)
                print(f"✓ Saved: {filename}")
                snapshot_count += 1

            time.sleep(0.01)

    finally:
        stream.stop()
        cv2.destroyAllWindows()


def view_both_cameras():
    """View both cameras side-by-side"""
    face_stream = CameraStream(9201, "Face Camera")
    chin_stream = CameraStream(9202, "Chin Camera")

    face_stream.start()
    chin_stream.start()

    print("\nViewing both cameras...")
    print("Press 'q' to quit, 's' to save snapshot")

    snapshot_count = 0
    last_face_frame = None
    last_chin_frame = None

    try:
        while True:
            face_frame = face_stream.get_frame()
            chin_frame = chin_stream.get_frame()

            if face_frame is not None:
                last_face_frame = face_frame
            if chin_frame is not None:
                last_chin_frame = chin_frame

            # Display side-by-side if we have both frames
            if last_face_frame is not None and last_chin_frame is not None:
                # Resize to same height if needed
                h1, w1 = last_face_frame.shape[:2]
                h2, w2 = last_chin_frame.shape[:2]

                if h1 != h2:
                    # Resize chin to match face height
                    scale = h1 / h2
                    new_w2 = int(w2 * scale)
                    last_chin_frame = cv2.resize(last_chin_frame, (new_w2, h1))

                # Concatenate horizontally
                combined = np.hstack([last_face_frame, last_chin_frame])

                # Add labels
                cv2.putText(combined, "Face Camera", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "Chin Camera", (w1 + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("GO1 - Both Cameras", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if last_face_frame is not None:
                    filename = f"face_snapshot_{snapshot_count}.jpg"
                    cv2.imwrite(filename, last_face_frame)
                    print(f"✓ Saved: {filename}")
                if last_chin_frame is not None:
                    filename = f"chin_snapshot_{snapshot_count}.jpg"
                    cv2.imwrite(filename, last_chin_frame)
                    print(f"✓ Saved: {filename}")
                snapshot_count += 1

            time.sleep(0.01)

    finally:
        face_stream.stop()
        chin_stream.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='GO1 Robot Camera Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 go1_camera_viewer.py --camera face    # View face camera
  python3 go1_camera_viewer.py --camera chin    # View chin camera  
  python3 go1_camera_viewer.py --camera both    # View both cameras

Make sure the GO1 Head Nano is streaming before running this!
        """
    )

    parser.add_argument(
        '--camera',
        choices=['face', 'chin', 'both'],
        default='both',
        help='Which camera(s) to view (default: both)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GO1 Camera Viewer")
    print("=" * 60)
    print(f"Receiving from: 192.168.123.13")
    print(f"Camera mode: {args.camera}")
    print("=" * 60)

    try:
        if args.camera == 'both':
            view_both_cameras()
        else:
            view_single_camera(args.camera)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()