import random
import numpy as np
import matplotlib.pyplot as plt


def line(x):
    return 3 * x + 2


def sgn(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1


class Perceptron:
    def __init__(self, dim, learning_rate, log_history=False):
        self.dim = dim
        # Na začatku se váhy + bias nastaví na náhodné hodnoty v rozsahu (-1,1  )
        self.w = [random.random() * 2 - 1 for _ in range(dim)]
        self.b = random.random() * 2 - 1
        self.learning_rate = learning_rate
        self.log_history = log_history
        self.history_w = []
        self.history_b = []
        self.history_pred = []

    def _train_one(self, x, y):
        # Predikce trénovacích dat pro výpočet chyby při současných parametrech (váhy+bias)
        y_guess = self._predict_one(x)
        error = y - y_guess
        # Úprava parametrů na základě chyby, vstupu a learning ratu
        for i in range(self.dim):
            self.w[i] += self.learning_rate * error * x[i]
        self.b += self.learning_rate * error

    def train(self, x, y, epochs):
        for _ in range(epochs):
            # Pro každé x z trénovací sady se provede úprava parametrů - tzn. batch_size = 1
            for i in range(len(x)):
                self._train_one(x[i], y[i])
            # Ukládání historie paramatrů a výstupů pro vizualizaci
            if self.log_history:
                self.history_w.append(self.w.copy())
                self.history_b.append(self.b)
                self.history_pred.append([self._predict_one(x_i) for x_i in x])

    def _predict_one(self, x):
        # Suma násobků vstupů a jim odpovídajících vah, ke kterým je přičten bias
        # ∑(x[i]*w[i]) + b
        y_guess = self.b
        for i in range(self.dim):
            y_guess += x[i] * self.w[i]
        # Výsledek predikce je získán z aktivační funkce
        return sgn(y_guess)

    def predict(self, x):
        return [self._predict_one(x_i) for x_i in x]


def main():
    # Generování n 2D bodů v rozsahu (-10,10)
    points = np.random.uniform(-10, 10, (100, 2))
    # Rozdělení vygenrovaných bodů na ty pod čarou a nad čarou
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
    # Vizualizace:
    # - Levý horní - n vygenerovaných bodů rozdělených funkcí y=3x+2
    # - Pravý horní - rozdělení vygenerovaných bodů pomocí natrénovaného perceptronu. Je také vidět predikovaná decision boundary
    # - Levý dolní - vývoj parametrů v jednotlivých epochách
    # - Pravý dolní - vývoj accuracy v jednotlivých epochách
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
    if p.w[1] != 0:
        y_vals = -(p.w[0] * x_vals + p.b) / p.w[1]
        axs[0, 1].plot(
            x_vals, y_vals, color="green", linestyle="--", label="Decision Boundary"
        )
    axs[0, 1].set_xlim(-10, 10)
    axs[0, 1].set_ylim(-10, 10)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].set_title("Predicted Data")
    axs[0, 1].legend()
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
