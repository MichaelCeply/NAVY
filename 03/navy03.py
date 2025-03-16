import tkinter as tk
import os
from matplotlib import cm, pyplot as plt
import numpy as np


def load_grid(filename):
    grid = []
    with open(filename, "r") as f:
        for line in f:
            row = [int(val) for val in line.split()]
            grid.append(row)
    return grid


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


class HopfieldNetwork:
    def __init__(self, grid_size):
        self.num_neurons = grid_size**2
        self.matrix = np.zeros((self.num_neurons, self.num_neurons))

    def train(self, patterns):
        # Naplneni matice nulami pri kazdem novem treninku
        self.matrix.fill(0)
        for pattern in patterns:
            pattern = np.array(pattern)
            # Zamena nul za -1
            pattern[pattern == 0] = -1
            # Prevod na 1D matici
            pattern = pattern.reshape((self.num_neurons, 1))
            # Maticovy soucit pro ziskani n^2 x n^2 matice vzoru
            w = pattern * pattern.T
            # Odstraneni napojeni samo na sebe - odstraneni hodnot na diagonale
            w = w - np.eye(self.num_neurons)
            # Pridani k dalsim vzorum
            self.matrix += w
        # Normalizace matice
        self.matrix /= len(patterns)

    def predict_synchronous(self, pattern, max_epoch=20):
        # Prevod na 1D matici
        pattern = np.array(pattern).reshape(self.num_neurons)
        # Zamena nul za -1
        pattern[pattern == 0] = -1
        for epoch in range(max_epoch):
            # Novy predikovany patern spocitany jako maticovy soucin
            # matice vzoru a postupne opravovaneho paternu,
            # ktery je prohnan funkci signum
            new_pattern = np.sign(self.matrix @ pattern)
            # Predcasne ukonceni, pokud nedoslo ke zmene
            if np.array_equal(new_pattern, pattern):
                break
            # Nahrazeni celeho opravovaneho paternu novym paternem
            pattern = new_pattern
        # Zamena -1 zpet na 0
        pattern[pattern == -1] = 0
        return pattern

    def predict_asynchronous(self, pattern, max_epoch=20):
        # Prevod na 1D matici
        pattern = np.array(pattern).reshape(self.num_neurons)
        # Zamena nul za -1
        pattern[pattern == 0] = -1
        for epoch in range(max_epoch):
            for i in range(self.num_neurons):
                # Postupne opravovani paternu pomoci skalarniho soucinu
                # i-teho radku matice vzoru a opravovaneho vzoru.
                # Vysledek opet prohnan funkci signum
                pattern[i] = sgn(np.dot(self.matrix[i, :], pattern))
        # Zamena -1 zpet na 0
        pattern[pattern == -1] = 0
        return pattern

    def plot_weights(self):
        # Vizualizace hodnot v matici vzoru
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.matrix, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.show()

    def restart(self):
        self.matrix = np.zeros((self.num_neurons, self.num_neurons))


class GridApp:
    def __init__(self, master, grid_size):
        self.master = master
        self.grid_size = grid_size
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype="int32")

        self.buttons = {}

        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.path = f"templates_{self.grid_size}"
        files = os.listdir(self.path)
        self.files = [file for file in files if file.endswith(".txt")]
        self.filenames = sorted([file[:-4] for file in self.files])

        self.clicked = tk.StringVar()
        self.clicked.set("-")

        self.hopfield = HopfieldNetwork(self.grid_size)
        self.hopfield.train([load_grid(f"{self.path}/{file}") for file in self.files])

        self.create_grid()
        self.create_controls()

    def create_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = "white"
                button = tk.Button(
                    self.grid_frame,
                    bg=color,
                    activebackground=color,
                    width=3,
                    height=2,
                    command=lambda r=r, c=c: self.on_click(r, c),
                )
                button.grid(row=r, column=c, padx=1, pady=1)
                self.buttons[(r, c)] = button

    def create_controls(self):
        drop = tk.OptionMenu(
            self.control_frame, self.clicked, *self.filenames, command=self.show_pattern
        )
        drop.pack(pady=5, fill=tk.X)

        save_btn = tk.Button(
            self.control_frame, text="Save Pattern", command=self.save_pattern
        )
        save_btn.pack(pady=5, fill=tk.X)

        clear_btn = tk.Button(
            self.control_frame, text="Clear Pattern", command=self.clear_pattern
        )
        clear_btn.pack(pady=5, fill=tk.X)

        sync_btn = tk.Button(self.control_frame, text="Sync", command=self.sync_pattern)
        sync_btn.pack(pady=5, fill=tk.X)

        async_btn = tk.Button(
            self.control_frame, text="Async", command=self.async_pattern
        )
        async_btn.pack(pady=5, fill=tk.X)

        weight_btn = tk.Button(
            self.control_frame, text="Show weights", command=self.show_weight
        )
        weight_btn.pack(pady=5, fill=tk.X)

    def on_click(self, r, c):
        button = self.buttons[(r, c)]
        current_color = button["bg"]
        new_color = "white" if current_color == "black" else "black"
        button.config(bg=new_color, activebackground=new_color)
        self.grid_data[r, c] = 1 if self.grid_data[r, c] == 0 else 0

    def show_pattern(self, value=None):
        self.grid_data = load_grid(f"{self.path}/{self.clicked.get()}.txt")
        self.update_grid()

    def save_pattern(self):
        with open(f"{self.path}/{self.clicked.get()}.txt", "w") as f:
            for r in range(self.grid_size):
                row_colors = []
                for c in range(self.grid_size):
                    color = self.buttons[(r, c)]["bg"]
                    row_colors.append("1" if color == "black" else "0")
                f.write(" ".join(row_colors) + "\n")
        print("Pattern saved to 'saved_pattern.txt'.")

    def clear_pattern(self):
        self.grid_data = np.zeros((self.grid_size, self.grid_size), dtype="int32")
        self.update_grid()

    def update_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = "black" if self.grid_data[r][c] == 1 else "white"
                self.buttons[(r, c)].config(bg=color, activebackground=color)

    def sync_pattern(self):
        self.hopfield.restart()
        self.hopfield.train([load_grid(f"{self.path}/{file}") for file in self.files])

        self.grid_data = self.hopfield.predict_synchronous(self.grid_data).reshape(
            (self.grid_size, self.grid_size)
        )

        self.update_grid()

    def async_pattern(self):
        self.hopfield.restart()
        self.hopfield.train([load_grid(f"{self.path}/{file}") for file in self.files])

        self.grid_data = self.hopfield.predict_asynchronous(self.grid_data).reshape(
            (self.grid_size, self.grid_size)
        )

        self.update_grid()

    def show_weight(self):
        self.hopfield.plot_weights()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridApp(root, 7)
    root.mainloop()
