from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk


class QLearningAgent(object):
    def __init__(
        self,
        grid_size,
        learning_rate=0.01,
        gamma=0.9,
        epsilon=0.1,
        cheese=(4, 4),
        start=(0, 0),
        walls=[],
        traps=[],
        epsilon_decay=False,
    ):
        # Velikost herni plochy
        self.grid_size = grid_size
        # Naplneni pameti agenta nulami
        self.q_matrix = np.zeros((4, grid_size, grid_size))
        # Mozne akce agenta
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Rychlost uceni
        self.learning_rate = learning_rate
        # Discount factor - urcuje jak moc agent dava vahu soucasnym odmenam pred budoucimi
        self.gamma = gamma
        # Exploration rate - mira jak moc bude agent prozkoumavat namisto vybirani nejlepsi moznosti
        self.epsilon = epsilon
        # Pozice startu,syra,zdi a pasti
        self.start = start
        self.cheese = cheese
        self.walls = walls
        self.traps = traps
        # nastaveni pro snizovani epsilonu aby se snizila explorace
        self.epsilon_decay = epsilon_decay

    def train(self, num_epochs):
        # Trenovani na poctu epoch
        for _ in range(num_epochs):
            # nastaveni soucasne pozice na startovni pozici na zacatku treninku
            state = self.start
            steps = 0
            # hledani dokud se nenajde syr nebo neubehne nejaky pocet kroku pro pripad zacykleni (napr neni mozna cesta k syru)
            while state != self.cheese and steps < 1000:
                # nahodny vyber mezi exploraci a nejlepsi moznosti
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(4)
                else:
                    action = np.argmax(self.q_matrix[:, state[0], state[1]])
                # vypocet nove pozice
                new_state = (
                    state[0] + self.actions[action][0],
                    state[1] + self.actions[action][1],
                )
                # kontrola zda agent nevyjel z herni plochy - pokud ano vrati se zpet na puvodni pozici a dostane pokutu 100
                if not (
                    0 <= new_state[0] < self.grid_size
                    and 0 <= new_state[1] < self.grid_size
                ):
                    reward = -100
                    new_state = state
                # kontrola kolize se zdi - funguje stejne jako vyjeti z pole
                elif new_state in self.walls:
                    reward = -100
                    new_state = state
                # kontrolo vstupu do pasti - udeli se pokuta 100, prepocita se q matrix pro akci a pozici a ukonci se generace
                elif new_state in self.traps:
                    reward = -100
                    self.q_matrix[action, state[0], state[1]] += self.learning_rate * (
                        reward
                        + self.gamma
                        * np.max(self.q_matrix[:, new_state[0], new_state[1]])
                        - self.q_matrix[action, state[0], state[1]]
                    )

                    break
                # nalezeni syru - odmena +100
                elif new_state == self.cheese:
                    reward = 100
                # krok na volne pole - zaporna hodnota pro minimalizaci poctu kroku
                else:
                    reward = -1
                # prepocitani q matice - k pozici rovne pozici agenta v q matice rovne zvolene akci se pricte max z
                # q matice pro vsechy akce
                old_state = state
                self.q_matrix[
                    action, old_state[0], old_state[1]
                ] += self.learning_rate * (
                    reward
                    + self.gamma * np.max(self.q_matrix[:, new_state[0], new_state[1]])
                    - self.q_matrix[action, old_state[0], old_state[1]]
                )
                # aktualizace stavu
                state = new_state
                steps += 1
                # snizovani epsilonu pro snizovani explorace
                if self.epsilon_decay:
                    self.epsilon *= 0.99

    def predict(self):
        state = self.start
        path = [state]
        steps = 0
        # hledani syru pomoci vybirani nejlepsi akce pro danou pozici
        while state != self.cheese and steps < 100:
            action = np.argmax(self.q_matrix[:, state[0], state[1]])
            new_state = (
                state[0] + self.actions[action][0],
                state[1] + self.actions[action][1],
            )
            # kontorola kolize se zdi a vyjetim z gridu
            if not (
                0 <= new_state[0] < self.grid_size
                and 0 <= new_state[1] < self.grid_size
            ):
                break
            if new_state in self.walls:
                break
            path.append(new_state)
            state = new_state
            steps += 1

        return path

    def visualize(self):
        # vizualizace q matrixu pro kazdou akci
        actions = ["Up", "Down", "Left", "Right"]
        vmin = np.min(self.q_matrix)
        vmax = np.max(self.q_matrix)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, ax in enumerate(axes):
            cax = ax.imshow(
                self.q_matrix[i],
                cmap="plasma",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Action: {actions[i]}")
            plt.colorbar(cax, ax=ax)
        plt.tight_layout()
        plt.show()


class GridApp:
    def __init__(self, master, grid_size):
        self.master = master
        self.grid_size = grid_size

        self.buttons = {}

        self.grid_frame = tk.Frame(self.master)
        self.grid_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.epochs_var = tk.IntVar(value=10000)
        self.epsilon_var = tk.IntVar(value=0.1)

        self.agent = QLearningAgent(self.grid_size, epsilon=0.5)
        self.mouse = (0, 0)
        self.cheese = (grid_size - 1, grid_size - 1)
        self.walls = set()
        self.traps = set()
        self.mouse_img = tk.PhotoImage(file="img/mouse.png")
        self.cheese_img = tk.PhotoImage(file="img/cheese.png")
        self.wall_img = tk.PhotoImage(file="img/wall.png")
        self.trap_img = tk.PhotoImage(file="img/trap.png")
        self.white_img = tk.PhotoImage(file="img/white.png")

        self.click_mode = None
        self.cell_size = 40

        self.create_grid()
        self.create_controls()

    def create_grid(self):
        for r in range(self.grid_size):
            self.grid_frame.grid_rowconfigure(r, weight=1, minsize=self.cell_size)
            for c in range(self.grid_size):
                self.grid_frame.grid_columnconfigure(
                    c, weight=1, minsize=self.cell_size
                )

                cell_frame = tk.Frame(
                    self.grid_frame, width=self.cell_size, height=self.cell_size
                )
                cell_frame.grid(row=r, column=c, sticky="nsew")
                cell_frame.grid_propagate(False)  # Prevent resizing

                if (r, c) == self.mouse:
                    image = self.mouse_img
                elif (r, c) == self.cheese:
                    image = self.cheese_img
                elif (r, c) in self.walls:
                    image = self.wall_img
                elif (r, c) in self.traps:
                    image = self.trap_img
                else:
                    image = self.white_img

                button = tk.Button(
                    cell_frame,
                    image=image,
                    command=lambda r=r, c=c: self.on_click(r, c),
                    bg="white",
                )
                button.pack(expand=True, fill="both")
                self.buttons[(r, c)] = button

    def create_controls(self):
        epochs_label = tk.Label(self.control_frame, text="Select epochs:")
        epochs_label.pack(pady=5)

        epochs_options = [1000, 10000, 100000]
        epochs_menu = tk.OptionMenu(
            self.control_frame, self.epochs_var, *epochs_options
        )
        epochs_menu.pack(pady=5, fill=tk.X)

        self.epsilon_scale = tk.Scale(
            self.control_frame,
            variable=self.epsilon_var,
            from_=0.0,
            to=1,
            digits=2,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            label="Set epsilon",
        )
        self.epsilon_scale.pack(pady=5)

        learn_btn = tk.Button(
            self.control_frame,
            text="Start learning",
            command=self.learn,
            bg="lime",
        )
        learn_btn.pack(pady=5, fill=tk.X)

        find_btn = tk.Button(
            self.control_frame,
            text="Find a cheese",
            command=self.find,
            bg="lime",
        )
        find_btn.pack(pady=5, fill=tk.X)

        mouse_btn = tk.Button(
            self.control_frame,
            text="Select a mouse",
            command=lambda: self.set_mode("mouse"),
            bg="yellow",
        )
        mouse_btn.pack(pady=5, fill=tk.X)

        trap_btn = tk.Button(
            self.control_frame,
            text="Select a trap",
            command=lambda: self.set_mode("trap"),
            bg="yellow",
        )
        trap_btn.pack(pady=5, fill=tk.X)

        wall_btn = tk.Button(
            self.control_frame,
            text="Select a wall",
            command=lambda: self.set_mode("wall"),
            bg="yellow",
        )
        wall_btn.pack(pady=5, fill=tk.X)

        cheese_btn = tk.Button(
            self.control_frame,
            text="Select a cheese",
            command=lambda: self.set_mode("cheese"),
            bg="yellow",
        )
        cheese_btn.pack(pady=5, fill=tk.X)

        remove_btn = tk.Button(
            self.control_frame,
            text="Clear a tile",
            command=lambda: self.set_mode("clear"),
            bg="red",
        )
        remove_btn.pack(pady=5, fill=tk.X)

        clear_btn = tk.Button(
            self.control_frame,
            text="Clear whole grid",
            command=self.reset,
            bg="red",
        )
        clear_btn.pack(pady=5, fill=tk.X)

        visualize_btn = tk.Button(
            self.control_frame,
            text="Visualize Q-matrix",
            command=self.visualize,
            bg="blue",
        )
        visualize_btn.pack(pady=5, fill=tk.X)

    def on_click(self, r, c):
        match self.click_mode:
            case "mouse":
                self.mouse = (r, c)
            case "wall":
                self.walls.add((r, c))
            case "trap":
                self.traps.add((r, c))
            case "cheese":
                self.cheese = (r, c)
            case "clear":
                if (r, c) == self.mouse:
                    self.mouse = (0, 0)
                elif (r, c) == self.cheese:
                    self.cheese = (self.grid_size - 1, self.grid_size - 1)
                elif (r, c) in self.walls:
                    self.walls.remove((r, c))
                elif (r, c) in self.traps:
                    self.traps.remove((r, c))
        self.update_grid()

    def learn(self):
        self.agent.epsilon = self.epsilon_var.get()
        self.agent.start = self.mouse
        self.agent.cheese = self.cheese
        self.agent.walls = self.walls
        self.agent.traps = self.traps
        self.agent.train(self.epochs_var.get())

    def find(self):
        path = self.agent.predict()
        if path[-1] == self.cheese:
            self.animate_path(path, 0)

    def animate_path(self, path, index):
        if index < len(path):
            self.mouse = path[index]
            self.update_grid()
            self.master.after(250, self.animate_path, path, index + 1)

    def set_mode(self, mode):
        self.click_mode = mode

    def reset(self):
        self.mouse = (0, 0)
        self.cheese = (self.grid_size - 1, self.grid_size - 1)
        self.walls = set()
        self.traps = set()
        self.update_grid()

    def visualize(self):
        self.agent.visualize()

    def update_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                button = self.buttons[(r, c)]
                if (r, c) == self.mouse:
                    button.configure(image=self.mouse_img, bg="white")
                elif (r, c) == self.cheese:
                    button.configure(image=self.cheese_img, bg="white")
                elif (r, c) in self.walls:
                    button.configure(image=self.wall_img, bg="white")
                elif (r, c) in self.traps:
                    button.configure(image=self.trap_img, bg="white")
                else:
                    button.configure(image=self.white_img, bg="white")


def main():
    root = tk.Tk()
    app = GridApp(root, 10)
    root.mainloop()


if __name__ == "__main__":
    main()
