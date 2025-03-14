import tkinter as tk

from matplotlib import cm, pyplot as plt
import numpy as np


def load_grid(filename):
    """Reads a grid configuration file and returns a 2D list of integers."""
    grid = []
    with open(filename, "r") as f:
        for line in f:
            # Convert each number in the line into an integer
            row = [int(val) for val in line.split()]
            grid.append(row)
    return grid


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.matrix = np.zeros((num_neurons**2, num_neurons**2))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            pattern[pattern == 0] = -1
            pattern = pattern.reshape((self.num_neurons**2, 1))
            w = pattern * pattern.T
            w = w - np.eye(self.num_neurons**2)
            self.matrix += w

    def predict_synchonouse(self, pattern, max_epoch=20):
        pattern = np.array(pattern).reshape(self.num_neurons**2, 1)
        pattern[pattern == 0] = -1
        for epoch in range(max_epoch):
            tmp_pattern = pattern.copy()
            for i in range(self.num_neurons):
                pattern[i] = sgn(np.dot(self.matrix[:, i], tmp_pattern).item)
        return pattern

    def predict_asynchonouse(self, pattern, max_epoch=20):
        pattern = np.array(pattern).reshape(self.num_neurons**2, 1)
        pattern[pattern == 0] = -1
        for epoch in range(max_epoch):
            for i in range(self.num_neurons**2):
                pattern[i] = sgn(np.dot(self.matrix[:, i], pattern).item())
        pattern[pattern == -1] = 0
        return pattern

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.matrix, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.show()


"""
class GridApp:
    def __init__(self, master, grid_data):
        self.master = master
        self.grid_data = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0]) if self.rows > 0 else 0

        # Dictionary to hold references to each button in the grid
        self.buttons = {}

        # Create two frames: one for the grid, one for the control buttons
        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create the grid of buttons
        self.create_grid()

        # Create the control buttons on the right
        self.create_controls()

    def create_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cell_value = self.grid_data[r][c]
                color = "black" if cell_value == 1 else "white"
                button = tk.Button(
                    self.grid_frame,
                    bg=color,
                    activebackground=color,  # Prevent color change on hover
                    width=3,  # Adjust size to your preference
                    height=2,  # Adjust size to your preference
                    command=lambda r=r, c=c: self.on_click(r, c),
                )
                button.grid(row=r, column=c, padx=1, pady=1)
                self.buttons[(r, c)] = button

    def create_controls(self):
        save_btn = tk.Button(
            self.control_frame, text="Save Pattern", command=self.save_pattern
        )
        save_btn.pack(pady=5, fill=tk.X)

        clear_btn = tk.Button(
            self.control_frame, text="Clear Pattern", command=self.clear_pattern
        )
        clear_btn.pack(pady=5, fill=tk.X)

        reconstruct_btn = tk.Button(
            self.control_frame, text="Reconstruct", command=self.reconstruct_pattern
        )
        reconstruct_btn.pack(pady=5, fill=tk.X)

    def on_click(self, r, c):
        button = self.buttons[(r, c)]
        current_color = button["bg"]
        new_color = "white" if current_color == "black" else "black"
        button.config(bg=new_color, activebackground=new_color)
        print(f"Button at ({r}, {c}) clicked. New color: {new_color}")

    def save_pattern(self):
        with open("saved_pattern.txt", "w") as f:
            for r in range(self.rows):
                row_colors = []
                for c in range(self.cols):
                    color = self.buttons[(r, c)]["bg"]
                    row_colors.append("1" if color == "black" else "0")
                f.write(" ".join(row_colors) + "\n")
        print("Pattern saved to 'saved_pattern.txt'.")

    def clear_pattern(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.buttons[(r, c)].config(bg="white", activebackground="white")
        print("Grid cleared.")

    def reconstruct_pattern(self):
        for r in range(self.rows):
            for c in range(self.cols):
                color = "black" if self.grid_data[r][c] == 1 else "white"
                self.buttons[(r, c)].config(bg=color, activebackground=color)
        print("Grid reconstructed from the original pattern.")

"""
if __name__ == "__main__":
    h = HopfieldNetwork(5)
    template1 = load_grid("01.txt")
    template2 = load_grid("02.txt")
    template3 = load_grid("03.txt")
    print(template1)
    h.train([template1, template2, template3])

    test1 = [
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]

    print(h.predict_asynchonouse(test1, 50))

    h.plot_weights()
    """
    # Load grid configuration from a file
    grid_data = load_grid("grid_config.txt")

    root = tk.Tk()
    root.title("Tkinter Grid with Controls")

    app = GridApp(root, grid_data)

    root.mainloop()
    """
