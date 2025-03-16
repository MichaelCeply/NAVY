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
    ):
        self.grid_size = grid_size
        self.q_matrix = np.zeros((4, grid_size, grid_size))
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.start = start
        self.cheese = cheese
        self.walls = walls
        self.traps = traps

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            state = self.start
            steps = 0
            while state != self.cheese and steps < 100:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(4)
                else:
                    action = np.argmax(self.q_matrix[:, state[0], state[1]])

                new_state = (
                    state[0] + self.actions[action][0],
                    state[1] + self.actions[action][1],
                )

                if not (
                    0 <= new_state[0] < self.grid_size
                    and 0 <= new_state[1] < self.grid_size
                ):
                    reward = -100
                    new_state = state
                elif new_state in self.walls:
                    reward = -100
                    new_state = state
                elif new_state in self.traps:
                    reward = -100
                    self.q_matrix[action, state[0], state[1]] += self.learning_rate * (
                        reward - self.q_matrix[action, state[0], state[1]]
                    )
                    break
                elif new_state == self.cheese:
                    reward = 100
                else:
                    reward = -1

                old_state = state
                self.q_matrix[
                    action, old_state[0], old_state[1]
                ] += self.learning_rate * (
                    reward
                    + self.gamma * np.max(self.q_matrix[:, new_state[0], new_state[1]])
                    - self.q_matrix[action, old_state[0], old_state[1]]
                )

                state = new_state
                steps += 1

    def predict(self):
        state = self.start
        path = [state]
        steps = 0

        while state != self.cheese and steps < 100:
            action = np.argmax(self.q_matrix[:, state[0], state[1]])
            new_state = (
                state[0] + self.actions[action][0],
                state[1] + self.actions[action][1],
            )
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

        self.agent = QLearningAgent(self.grid_size)
        self.mouse = (0, 0)
        self.cheese = (grid_size - 1, grid_size - 1)
        self.walls = []
        self.traps = []
        self.mouse_img = tk.PhotoImage(file="img/mouse.png")
        self.cheese_img = tk.PhotoImage(file="img/cheese.png")
        self.wall_img = tk.PhotoImage(file="img/wall.png")
        self.trap_img = tk.PhotoImage(file="img/trap.png")

        self.cell_size = 40

        self.create_grid()
        self.create_controls()

    def create_grid(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) == self.mouse:
                    image = self.mouse_img
                elif (r, c) == self.cheese:
                    image = self.cheese_img
                elif (r, c) in self.walls:
                    image = self.wall_img
                elif (r, c) in self.traps:
                    image = self.trap_img
                else:
                    image = None

                button = tk.Button(
                    self.grid_frame,
                    image=image,
                    width=3,
                    height=2,
                    command=lambda r=r, c=c: self.on_click(r, c),
                    bg="white" if image is None else None,
                    activebackground="white" if image is None else None,
                )

                button.grid(row=r, column=c, padx=0, pady=0, sticky="nsew")
                self.buttons[(r, c)] = button

        for i in range(self.grid_size):
            self.grid_frame.grid_rowconfigure(i, weight=1)
            self.grid_frame.grid_columnconfigure(i, weight=1)

    def create_controls(self):
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
            command=self.learn,
            bg="yellow",
        )
        mouse_btn.pack(pady=5, fill=tk.X)

        trap_btn = tk.Button(
            self.control_frame,
            text="Select a trap",
            command=self.learn,
            bg="yellow",
        )
        trap_btn.pack(pady=5, fill=tk.X)

        wall_btn = tk.Button(
            self.control_frame,
            text="Select a wall",
            command=self.learn,
            bg="yellow",
        )
        wall_btn.pack(pady=5, fill=tk.X)

        cheese_btn = tk.Button(
            self.control_frame,
            text="Select a cheese",
            command=self.learn,
            bg="yellow",
        )
        cheese_btn.pack(pady=5, fill=tk.X)

        remove_btn = tk.Button(
            self.control_frame,
            text="Clear a tile",
            command=self.learn,
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

    def on_click(self, r, c):
        print(f"Button at ({r}, {c}) clicked!")

    def learn(self):
        self.agent.start = self.mouse
        self.agent.cheese = self.cheese
        self.agent.walls = self.walls
        self.agent.traps = self.traps
        self.agent.train(10_000)

    def find(self):
        path = self.agent.predict()
        self.animate_path(path, 0)

    def animate_path(self, path, index):
        if index < len(path):
            self.mouse = path[index]
            self.update_grid()
            self.master.after(500, self.animate_path, path, index + 1)

    def reset(self):
        self.mouse = (0, 0)
        self.cheese = (self.grid_size - 1, self.grid_size - 1)
        self.walls = []
        self.traps = []
        self.update_grid()

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
                    button.configure(image="", bg="white")


def main():
    root = tk.Tk()
    app = GridApp(root, 10)
    root.mainloop()


if __name__ == "__main__":
    main()
