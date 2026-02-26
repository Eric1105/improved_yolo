import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def resolve_weights_path(p: str) -> str:
    """Handle common path typo: 'home/xxx' -> '/home/xxx'."""
    path = Path(p)
    if path.exists():
        return str(path)
    if not p.startswith("/"):
        p2 = "/" + p
        if Path(p2).exists():
            return p2
    raise FileNotFoundError(f"Weights not found: {p} (also tried: /{p})")


def parse_imgsz(imgsz_list):
    """
    Accept:
      --imgsz 1024
      --imgsz 1824 2752
    Return int or (h, w).
    """
    if len(imgsz_list) == 1:
        return int(imgsz_list[0])
    if len(imgsz_list) == 2:
        h, w = int(imgsz_list[0]), int(imgsz_list[1])
        return (h, w)
    raise ValueError("--imgsz must be one int (square) or two ints (h w).")


def draw_detections(frame, boxes_xyxy, confs, clss, names=None, conf_thres=0.1):
    """
    Draw detections on frame (in-place).
    boxes_xyxy: Nx4 float, in original image coordinates
    confs: N float
    clss: N float/int
    names: dict/list mapping class id -> name
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return frame

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, clss):
        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        if names is None:
            label = f"{cls_id} {conf:.2f}"
        else:
            if isinstance(names, dict):
                cname = names.get(cls_id, str(cls_id))
            else:  # list
                cname = names[cls_id] if cls_id < len(names) else str(cls_id)
            label = f"{cname} {conf:.2f}"

        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True, help="input video path")
    ap.add_argument("--out", type=str, default="output_annotated.mp4", help="output video path")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1024], help="imgsz or (h w)")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--max_det", type=int, default=3000)
    ap.add_argument("--show", action="store_true", help="show preview window")
    args = ap.parse_args()

    weights = resolve_weights_path(args.weights)
    imgsz = parse_imgsz(args.imgsz)

    model = YOLO(weights)

    # names mapping (best-effort for older versions)
    names = None
    try:
        names = model.model.names
    except Exception:
        try:
            names = model.names
        except Exception:
            names = None

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {args.out}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Predict on this frame
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )

        r = results[0]

        # Extract boxes
        boxes_xyxy = None
        confs = None
        clss = None
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=np.float32)

        annotated = frame.copy()
        annotated = draw_detections(
            annotated, boxes_xyxy, confs, clss, names=names, conf_thres=args.conf
        )

        out.write(annotated)

        if args.show:
            cv2.imshow("predict", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done. Saved annotated video to: {args.out}")


if __name__ == "__main__":
    main()
