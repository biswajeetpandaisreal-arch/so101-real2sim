"""Live preview of a USB camera (q to close).

    python scripts/camera_preview.py --index 1
"""

import argparse

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frame")
            break
        cv2.imshow(f"camera {a.index} (q to close)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
