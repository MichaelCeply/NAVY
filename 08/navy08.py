import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, RadioButtons
import matplotlib.colors as colors


class Fractal:
    def __init__(self):
        # hranice pro Mandelbrot
        self.mandelbrot_x_min, self.mandelbrot_x_max = -2.0, 1.0
        self.mandelbrot_y_min, self.mandelbrot_y_max = -1.5, 1.5

        # hranice pro Julia
        self.julia_x_min, self.julia_x_max = -1.5, 1.5
        self.julia_y_min, self.julia_y_max = -1.5, 1.5

        # default
        self.fractal_type = "Mandelbrot"
        self.set_default_view()

        # const param pro Julia
        self.julia_c = complex(-0.7, 0.27)

        # params pro vypocet
        self.max_iterations = 100
        self.escape_radius = 2.0

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95)

        self.fractal_image = None
        self.plot_fractal()

        # zoom
        self.rs = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        self.setup_ui()

        plt.show()

    # vychozi nastaveni pro zobrazeni
    def set_default_view(self):
        if self.fractal_type == "Mandelbrot":
            self.x_min, self.x_max = self.mandelbrot_x_min, self.mandelbrot_x_max
            self.y_min, self.y_max = self.mandelbrot_y_min, self.mandelbrot_y_max
        else:
            self.x_min, self.x_max = self.julia_x_min, self.julia_x_max
            self.y_min, self.y_max = self.julia_y_min, self.julia_y_max

    def setup_ui(self):
        reset_ax = plt.axes([0.8, 0.05, 0.15, 0.04])
        self.reset_button = Button(reset_ax, "Reset")
        self.reset_button.on_clicked(self.reset_view)

        radio_ax = plt.axes([0.05, 0.05, 0.15, 0.1])
        self.radio = RadioButtons(radio_ax, ("Mandelbrot", "Julia"))
        self.radio.on_clicked(self.set_fractal_type)

    # vypocet - mandelbrot
    def compute_mandelbrot(self, h, w):
        x = np.linspace(self.x_min, self.x_max, w)
        y = np.linspace(self.y_min, self.y_max, h)
        C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

        Z = np.zeros_like(C, dtype=complex)
        iterations = np.zeros(C.shape, dtype=int)
        mask = np.ones(C.shape, dtype=bool)

        for i in range(self.max_iterations):
            Z[mask] = Z[mask] ** 2 + C[mask]
            diverged = np.abs(Z) > self.escape_radius
            iterations[diverged & mask] = i
            mask[diverged] = False
            if not np.any(mask):
                break

        return iterations

    # vypocet - julia
    def compute_julia(self, h, w):
        x = np.linspace(self.x_min, self.x_max, w)
        y = np.linspace(self.y_min, self.y_max, h)
        Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

        iterations = np.zeros(Z.shape, dtype=int)
        mask = np.ones(Z.shape, dtype=bool)

        for i in range(self.max_iterations):
            Z[mask] = Z[mask] ** 2 + self.julia_c
            diverged = np.abs(Z) > self.escape_radius
            iterations[diverged & mask] = i
            mask[diverged] = False
            if not np.any(mask):
                break

        return iterations

    def plot_fractal(self):
        h, w = 800, 1000

        if self.fractal_type == "Mandelbrot":
            iterations = self.compute_mandelbrot(h, w)
            title = "Mandelbrot"
        else:
            iterations = self.compute_julia(h, w)
            title = f"Julia"

        norm = colors.PowerNorm(0.3)

        if self.fractal_image is None:
            self.fractal_image = self.ax.imshow(
                iterations,
                cmap="hot",
                norm=norm,
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                origin="lower",
                interpolation="bicubic",
            )
        else:
            self.fractal_image.set_data(iterations)
            self.fractal_image.set_extent(
                [self.x_min, self.x_max, self.y_min, self.y_max]
            )

        self.ax.set_xlabel("Re")
        self.ax.set_ylabel("Im")
        self.ax.set_title(f"{title}\nIteration: {self.max_iterations}")
        self.fig.canvas.draw_idle()

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        self.x_min, self.x_max = min(x1, x2), max(x1, x2)
        self.y_min, self.y_max = min(y1, y2), max(y1, y2)

        self.plot_fractal()

    def reset_view(self, event):
        self.set_default_view()
        self.plot_fractal()

    def set_fractal_type(self, label):
        if label != self.fractal_type:
            self.fractal_type = label
            self.set_default_view()
            plt.clf()
            self.ax = self.fig.add_subplot(111)
            self.fractal_image = None
            self.rs = RectangleSelector(
                self.ax,
                self.on_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
            )
            self.setup_ui()
            self.plot_fractal()


if __name__ == "__main__":
    explorer = Fractal()
