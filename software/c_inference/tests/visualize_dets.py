import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO detections")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.image}")

    height, width = image.shape[:2]

    with open(args.names) as f:
        names = [line.strip() for line in f]

    with open(args.detections) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            cls, prob, cx, cy, bw, bh = line.split()
            cls = int(cls)
            prob = float(prob)
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

            if not 0 <= cls < len(names):
                raise ValueError(f"Invalid class index: {cls}")

            x1 = max(0, min(int((cx - bw / 2) * width), width - 1))
            y1 = max(0, min(int((cy - bh / 2) * height), height - 1))
            x2 = max(0, min(int((cx + bw / 2) * width), width - 1))
            y2 = max(0, min(int((cy + bh / 2) * height), height - 1))

            label = f"{names[cls]} {prob * 100:.1f}%"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y1, th + 8)

            cv2.rectangle(
                image, (x1, label_y - th - 8),
                (x1 + tw + 6, label_y), (0, 255, 0), -1
            )
            cv2.putText(
                image, label, (x1 + 3, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Cannot save output image: {output}")

    print(f"Saved visualization: {output}")


if __name__ == "__main__":
    main()
