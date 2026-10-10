"""
Drowsiness & Yawn Detection (driver monitoring)
Watches a face via MediaPipe FaceMesh and raises an alert when the eyes stay
closed too long (drowsiness, via Eye Aspect Ratio) or a yawn is detected
(via Mouth Aspect Ratio) — the same idea behind automotive driver-monitoring
systems. Runs fully offline, no training or model download required.

Usage:
    python drowsiness_detector.py --source 0
    python drowsiness_detector.py --source dashcam.mp4
"""

import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = (13, 14)


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe drowsiness + yawn detector")
    parser.add_argument("--source", default="rtsp://avalos:12345678@192.168.1.3:8080/h264_ulaw.sdp", help="Video path or webcam index (default: 0)")
    parser.add_argument("--ear-thresh", type=float, default=0.21, help="Eye Aspect Ratio below this counts as 'closed'")
    parser.add_argument("--ear-frames", type=int, default=20, help="Consecutive closed-eye frames before a drowsiness alert")
    parser.add_argument("--mar-thresh", type=float, default=0.6, help="Mouth Aspect Ratio above this counts as a yawn")
    parser.add_argument("--mute", action="store_true", help="Disable the audio alarm")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    parser.add_argument("--front-camera", action="store_true", default=True, help="Attempt to switch to the front camera (IP Webcam app only)")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4) * 2
    return vertical / horizontal if horizontal else 0.0


def mouth_aspect_ratio(landmarks, w, h):
    lx, ly = landmarks[MOUTH_CORNERS[0]]
    rx, ry = landmarks[MOUTH_CORNERS[1]]
    ux, uy = landmarks[MOUTH_VERTICAL[0]]
    dx, dy = landmarks[MOUTH_VERTICAL[1]]
    horizontal = np.hypot(rx - lx, ry - ly)
    vertical = np.hypot(dx - ux, dy - uy)
    return vertical / horizontal if horizontal else 0.0


def beep():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1200, 300)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def main():
    args = parse_args()

    if args.front_camera and args.source.startswith("rtsp://"):
        import urllib.request
        try:
            from urllib.parse import urlparse
            parsed = urlparse(args.source)
            url = f"http://{parsed.netloc}/settings/ffc?set=on"
            print(f"Switching to front camera via {url}")
            urllib.request.urlopen(url, timeout=3)
            time.sleep(1) # wait for the camera to switch
        except Exception as e:
            print(f"Could not automatically switch to front camera: {e}")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import os
    if not os.path.exists("face_landmarker.task"):
        import urllib.request
        print("Downloading face_landmarker.task...")
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')

    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    closed_frames = 0
    last_beep = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            points = np.array([(p.x * w, p.y * h) for p in lm])

            left_ear = eye_aspect_ratio(points[LEFT_EYE])
            right_ear = eye_aspect_ratio(points[RIGHT_EYE])
            ear = (left_ear + right_ear) / 2
            mar = mouth_aspect_ratio(points, w, h)

            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, tuple(points[idx].astype(int)), 1, (0, 255, 0), -1)

            drowsy = False
            if ear < args.ear_thresh:
                closed_frames += 1
                if closed_frames >= args.ear_frames:
                    drowsy = True
            else:
                closed_frames = 0

            yawning = mar > args.mar_thresh

            cv2.putText(frame, f"EAR: {ear:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if drowsy:
                cv2.putText(frame, "DROWSINESS ALERT", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                now = time.time()
                if not args.mute and now - last_beep > 1.5:
                    last_beep = now
                    beep()
            elif yawning:
                cv2.putText(frame, "YAWN DETECTED", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
        else:
            cv2.putText(frame, "No face detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not args.no_show:
            cv2.imshow("Drowsiness & Yawn Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
