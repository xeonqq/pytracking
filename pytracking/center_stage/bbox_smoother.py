import argparse
import matplotlib.pyplot as plt


class SecondOrderFilter:

    def __init__(self, tau, dt, zeta=1.0):
        """
        Initialize a second-order filter using backward Euler.

        Args:
            tau (float): Time constant.
            dt (float): Sampling interval.
            zeta (float): Damping ratio (default: 1.0 for critical damping).
        """
        self.tau = tau
        self.dt = dt
        self.zeta = zeta

        # Precompute coefficients
        self.a0 = tau**2 / dt**2 + 2 * zeta * tau / dt + 1
        self.a1 = -2 * tau**2 / dt**2 + 2
        self.a2 = tau**2 / dt**2 - 2 * zeta * tau / dt + 1

        self.b0 = 1 / self.a0
        self.b1 = 2 / self.a0
        self.b2 = 1 / self.a0

        # Internal state
        self.x1, self.x2 = 0.0, 0.0  # previous two inputs
        self.y1, self.y2 = 0.0, 0.0  # previous two outputs

    def reset(self, initial_value):
        self.x1 = self.x2 = initial_value
        self.y1 = self.y2 = initial_value

    def update(self, x0):
        """
        Process one input sample and return the filtered output.

        Args:
            x0 (float): Current input value.

        Returns:
            float: Filtered output.
        """
        y0 = (self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2 -
              (self.a1 / self.a0) * self.y1 - (self.a2 / self.a0) * self.y2)

        # Update state
        self.x2, self.x1 = self.x1, x0
        self.y2, self.y1 = self.y1, y0

        return y0


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


class BBoxSmootherPt2:

    def __init__(self, tau=0.1, dt=1 / 30.0, zeta=1.0):
        """
        Bounding box smoother using second-order filters.

        Args:
            tau (float): Time constant for smoothing.
            dt (float): Sampling interval (1 / frame rate).
            zeta (float): Damping ratio (default: 1.0 = critically damped).
        """
        self.filter_x = SecondOrderFilter(tau, dt, zeta)
        self.filter_y = SecondOrderFilter(tau, dt, zeta)
        self.filter_w = SecondOrderFilter(tau, dt, zeta)
        self.filter_h = SecondOrderFilter(tau, dt, zeta)
        self.first = True

    def apply(self, x, y, w, h):
        """
        Apply the smoother to a new bounding box.

        Returns:
            Tuple of smoothed (x, y, w, h)
        """
        if self.first:
            # Initialize filters with the first value by feeding it twice
            self.filter_x.reset(x)
            self.filter_y.reset(y)
            self.filter_w.reset(w)
            self.filter_h.reset(h)
            self.first = False

        return (self.filter_x.update(x), self.filter_y.update(y),
                self.filter_w.update(w), self.filter_h.update(h))


def read_bounding_boxes(filename):
    centers = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # skip malformed lines
            try:
                x, y, w, h = map(float, parts)
                cx = x + w / 2
                cy = y + h / 2
                centers.append((cx, cy))
            except ValueError:
                continue  # skip lines with invalid numbers
    return centers


def plot_centers(centers, c='blue'):
    if not centers:
        print("No valid bounding boxes found in the file.")
        return
    xs, ys = zip(*centers)
    plt.scatter(xs, ys, c=c, marker='x')
    plt.title('Centers of Bounding Boxes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')


def main():
    parser = argparse.ArgumentParser(
        description=
        "Plot the centers of bounding boxes from a file. Each row in the file should be 'x, y, w, h'."
    )
    parser.add_argument('filename',
                        type=str,
                        help="Path to the file containing the bounding boxes")
    parser.add_argument(
        '--smooth_alpha',
        type=float,
        default=0.25,
        help=
        'Smoothing factor (0.0-1.0), lower=more smoothing when using moving average, it meaning time constant when using PT2 filter, larger the more smooth'
    )

    args = parser.parse_args()

    centers = read_bounding_boxes(args.filename)

    smoother = BBoxSmoother(alpha=args.smooth_alpha)
    smoother_pt2 = BBoxSmootherPt2(tau=0.1, dt=1 / 24.0)
    smoothed_centers = []
    smoothed_centers_pt2 = []
    for i, (cx, cy) in enumerate(centers):
        # Assuming w and h are 10 for the sake of smoothing
        smoothed_x, smoothed_y, _, _ = smoother.apply(cx, cy, 10, 10)
        smoothed_centers.append((smoothed_x, smoothed_y))

        smoothed_x, smoothed_y, _, _ = smoother_pt2.apply(cx, cy, 10, 10)

        smoothed_centers_pt2.append((smoothed_x, smoothed_y))
    plot_centers(centers)
    plot_centers(smoothed_centers, c='red')
    plot_centers(smoothed_centers_pt2, c='cyan')
    plt.show()


if __name__ == '__main__':
    main()
