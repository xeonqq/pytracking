import cv2
import argparse
from super_res import SuperResEngine, Resizer


class BBoxSmoother:

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None
        self.prev_w = None
        self.prev_h = None

    def apply(self, x, y, w, h):
        if self.prev_x is None:  # First frame
            self.prev_x = x
            self.prev_y = y
            self.prev_w = w
            self.prev_h = h
            return x, y, w, h

        # Apply exponential moving average
        smoothed_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.prev_y
        smoothed_w = self.alpha * w + (1 - self.alpha) * self.prev_w
        smoothed_h = self.alpha * h + (1 - self.alpha) * self.prev_h

        # Update previous values
        self.prev_x = smoothed_x
        self.prev_y = smoothed_y
        self.prev_w = smoothed_w
        self.prev_h = smoothed_h

        return smoothed_x, smoothed_y, smoothed_w, smoothed_h


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def main():
    parser = argparse.ArgumentParser(description='Center Stage Video Cropping')
    parser.add_argument('--input_video',
                        required=True,
                        help='Input video file path')
    parser.add_argument('--bbox_file',
                        required=True,
                        help='Bounding box text file path')
    parser.add_argument('--output_video',
                        required=True,
                        help='Output video file path')
    parser.add_argument(
        '--smooth_alpha',
        type=float,
        default=0.25,
        help='Smoothing factor (0.0-1.0), lower=more smoothing')

    parser.add_argument(
        '--sr_model_path',
        type=str,
        help=
        'Path to the super-resolution model, i.e. ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    )
    args = parser.parse_args()
    if args.sr_model_path:
        super_res_engine = SuperResEngine(args.sr_model_path)
    else:
        super_res_engine = Resizer()

    # Video input setup
    cap = cv2.VideoCapture(args.input_video)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    aspect_ratio = orig_width / orig_height

    # Output video setup (fixed size 200px width)
    output_width = 1280
    output_height = int(output_width / aspect_ratio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps,
                          (output_width, output_height))

    # Read bounding box data
    with open(args.bbox_file, 'r') as f:
        bboxes = [list(map(float, line.strip().split())) for line in f]

    smoother = BBoxSmoother(alpha=args.smooth_alpha)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= len(bboxes):
            break

        raw_x, raw_y, raw_w, raw_h = bboxes[frame_count]
        x, y, w, h = smoother.apply(raw_x, raw_y, raw_w, raw_h)
        x, y, w, h = int(x), int(y), int(w), int(h)
        frame_count += 1
        print(f"Frame {frame_count}: x={x}, y={y}, w={w}, h={h}")

        # Calculate minimum crop width
        crop_height = max(w, h) * 4
        crop_width = int(crop_height * aspect_ratio)

        # Calculate center coordinates
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate crop boundaries
        crop_x1 = center_x - crop_width // 2
        crop_y1 = center_y - crop_height // 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # Clamp coordinates to video dimensions
        crop_x1 = clamp(crop_x1, 0, orig_width)
        crop_y1 = clamp(crop_y1, 0, orig_height)
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        crop_x2 = clamp(crop_x2, 0, orig_width)
        crop_y2 = clamp(crop_y2, 0, orig_height)
        # print(f"Frame {frame_count}: {crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}")

        # Crop and resize
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        resized = super_res_engine.apply(cropped,
                                         (output_width, output_height))

        # Write output frame
        out.write(resized)

    cap.release()
    out.release()
    print(f"Processing completed. Output video saved to {args.output_video}")


if __name__ == "__main__":
    main()
