import random
import numpy as np
import matplotlib.pyplot as plt


def sgn(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1


def line(x):
    return 3 * x + 2


class Perceptron:
    def __init__(self, dim, learning_rate, log_history=False):
        self.dim = dim
        self.w = [random.random() * 2 - 1] * dim
        self.b = random.random() * 2 - 1
        self.learning_rate = learning_rate
        self.log_history = log_history
        self.history_w = []
        self.history_b = []
        self.history_pred = []

    def train_one(self, x, y):
        y_guess = self.predict_one(x)
        error = y - y_guess
        for i in range(self.dim):
            self.w[i] = self.w[i] + self.learning_rate * error * x[i]
        self.b = self.b + self.learning_rate * error

    def train(self, x, y, epochs):
        for _ in range(epochs):
            for i in range(len(x)):
                self.train_one(x[i], y[i])
            if self.log_history:
                self.history_w.append(self.w.copy())
                self.history_b.append(self.b)
                self.history_pred.append([self.predict_one(x_i) for x_i in x])

    def predict_one(self, x):
        y_guess = self.b
        for i in range(self.dim):
            y_guess += x[i] * self.w[i]
        return sgn(y_guess)

    def predict(self, x):
        return [self.predict_one(x_i) for x_i in x]


def main():
    points = np.random.uniform(-10, 10, (500, 2))

    y_train = []
    for point in points:
        x, y = point
        y_train.append(sgn(y - line(x)))

    p = Perceptron(2, 0.0001, log_history=True)
    p.train(points, y_train, 100)
    y_pred = p.predict(points)

    accuracies = []
    for history in p.history_pred:
        acc = sum(1 for pred, true in zip(history, y_train) if pred == true) / len(
            y_train
        )
        accuracies.append(acc)
    iterations = range(len(accuracies))

    x_vals = np.linspace(-10, 10, 10000)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].plot(x_vals, line(x_vals), color="black", label="y = 3x + 2")

    for i in range(len(points)):
        x, y = points[i]
        axs[0, 0].scatter(x, y, color="red" if y_train[i] == 1 else "blue")
        axs[0, 1].scatter(x, y, color="red" if y_pred[i] == 1 else "blue")

    axs[0, 0].set_xlim(-10, 10)
    axs[0, 0].set_ylim(-10, 10)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].set_title("Generated Data")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(x_vals, line(x_vals), color="black", label="y = 3x + 2")
    axs[0, 1].set_xlim(-10, 10)
    axs[0, 1].set_ylim(-10, 10)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].set_title("Predicted Data")
    axs[0, 1].grid(True)

    history_w = np.array(p.history_w)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(p.dim):
        axs[1, 0].plot(
            iterations,
            history_w[:, i],
            label=f"w[{i}]",
            color=colors[i % len(colors)],
        )
    axs[1, 0].plot(
        iterations, p.history_b, label="b", color=colors[p.dim % len(colors)]
    )
    axs[1, 0].set_title("Weights and Bias")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(iterations, accuracies, color="magenta")
    axs[1, 1].set_title("Accuracy")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
