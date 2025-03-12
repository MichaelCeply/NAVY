import random
import math

from matplotlib import pyplot as plt
import numpy as np

NUM_OF_SEPARATORS = 100


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def calculate_error(y_train, y_pred):
    sum = 0
    for train, pred in zip(y_train, y_pred):
        sum += 1 / 2 * (train - pred) ** 2
    return sum


def compare(x, y_train, y_pred):
    print("-" * NUM_OF_SEPARATORS + "\nResults:")
    for i in range(len(x)):
        pred_str = ", ".join(f"{round(p, 4)}" for p in y_pred[i])
        train_str = ", ".join(f"{yt}" for yt in y_train[i])
        print(f"Input: {x[i]} -> Predicted: [{pred_str}] | Expected: [{train_str}]")
    print("-" * NUM_OF_SEPARATORS)


class Perceptron:
    def __init__(self, dim: int, name: str, learning_rate: float):
        # Na začatku se váhy + bias nastaví na náhodné hodnoty v rozsahu (-1,1  )
        self.dim = dim
        self.w = [random.random() * 2 - 1 for _ in range(dim)]
        self.b = random.random() * 2 - 1
        self.name = name
        self.learning_rate = learning_rate
        self.output = 0
        self.delta = 0
        self.error = 0
        self.history_w = []
        self.history_b = []
        self.history_output = []

    def activate(self, x: list) -> float:
        # Suma násobků vstupů a jim odpovídajících vah, ke kterým je přičten bias
        # ∑(x[i]*w[i]) + b
        output = self.b
        for i in range(self.dim):
            output += x[i] * self.w[i]
        # Výsledek predikce je získán z aktivační funkce
        self.output = sigmoid(output)
        return self.output

    def __str__(self):
        return f"{self.name}; w: {self.w}; b: {self.b}"


class Layer:
    # Třída reprezentující jednu vrstvu v neuronové síti
    def __init__(self, num_perceptors, name, dim, learing_rate):
        self.perceptrons = [
            Perceptron(dim, f"{name}_p_{i}", learing_rate)
            for i in range(num_perceptors)
        ]
        self.name = name

    def predict(self, x):
        return [perceptron.activate(x) for perceptron in self.perceptrons]

    def __str__(self):
        per_str = "\n".join(str(per) for per in self.perceptrons)
        return f"{self.name}:\n{per_str}"


class MLP:
    def __init__(self, dim, layer_sizes, learning_rate, log_history=False):
        # Inicializace ANN podle architektury v layer_sizes
        self.dim = dim
        self.layers: list[Layer] = []
        for i in range(len(layer_sizes)):
            name = "hidden" if i < len(layer_sizes) - 1 else "output"
            l_dim = dim if i == 0 else layer_sizes[i - 1]
            self.layers.append(Layer(layer_sizes[i], name, l_dim, learning_rate))

        self.total_error_history = []
        self.train_data = []
        self.prediction_history = []

    def train(self, x, y, epochs):
        self.train_data = x
        for epoch in range(epochs):
            # Pro každý vstup je vypočítána chyba
            total_error = 0
            epoch_pred = []
            for x_i, y_i in zip(x, y):
                activations = [x_i]
                for layer in self.layers:
                    activations.append(layer.predict(activations[-1]))

                y_pred = activations[-1]
                epoch_pred.append(y_pred)
                total_error += calculate_error(y_i, y_pred)
                # Back propagation
                # Pro výstupní vrstvu se delta vah spočítá jako rozdíl mezi vstupem a predikcí krát derivace aktivační funkce s predikcí
                output_layer = self.layers[-1]
                for k, perceptron in enumerate(output_layer.perceptrons):
                    perceptron.delta = (y_pred[k] - y_i[k]) * sigmoid_derivative(
                        y_pred[k]
                    )
                # Pro skryté vrsty provádím back propagation odzadu(od výstupní vrstvy)
                # deltu vah spočítáme jako součet součinů vah a delt, který vynásobíme derivací aktivační funkce s predikcí určeného neuronu
                for i in range(len(self.layers) - 2, -1, -1):
                    current_layer = self.layers[i]
                    next_layer = self.layers[i + 1]
                    for j, perceptron in enumerate(current_layer.perceptrons):
                        error = sum(
                            next_p.delta * next_p.w[j]
                            for next_p in next_layer.perceptrons
                        )
                        perceptron.delta = error * sigmoid_derivative(perceptron.output)
                # Úprava parametrů na základě chyby, vstupu a delty
                for i, layer in enumerate(self.layers):
                    inputs = activations[i]
                    for perceptron in layer.perceptrons:
                        for k in range(perceptron.dim):
                            perceptron.w[k] -= (
                                perceptron.learning_rate * perceptron.delta * inputs[k]
                            )
                        perceptron.b -= perceptron.learning_rate * perceptron.delta
            # Ukládání historie paramatrů a výstupů pro vizualizaci
            self.total_error_history.append(total_error)
            self.prediction_history.append(epoch_pred)
            for layer in self.layers:
                for perceptron in layer.perceptrons:
                    perceptron.history_w.append(perceptron.w.copy())
                    perceptron.history_b.append(perceptron.b)

            if epoch % (epochs / 10) == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")

    def predict(self, x):
        # Postupné predikování jednotlivých vrstev - predikce vrstvy 1 je vstupem pro vrstvu 2
        result = []
        for x_i in x:
            layer_input = x_i
            for layer in self.layers:
                layer_input = layer.predict(layer_input)
            result.append(layer_input)
        return result

    def visualize(self):
        # Funkce pro vizalizaci učení
        # Vlevo nahoře - vývoj součtu chyb v čase
        # Vpravo nahoře - vývoj predikcí pro jednolivé trénovací data
        # Vlevo dole - vývoj parametrů v jednotlivých epochách
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        axs[0, 0].plot(self.total_error_history, label="Total Error")
        axs[0, 0].set_title("Total error")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        transposed = list(zip(*self.prediction_history))
        for i, y_values in enumerate(transposed):
            axs[0, 1].plot(y_values, label=f"Prediction for {self.train_data[i]}")
        axs[0, 1].set_title("Predicted values")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        for layer in self.layers:
            for perceptron in layer.perceptrons:
                for i in range(perceptron.dim):
                    history_w = np.array(perceptron.history_w)
                    axs[1, 0].plot(history_w[:, i], label=f"{perceptron.name}_w{i}")
                axs[1, 0].plot(perceptron.history_b, label=f"{perceptron.name}_b")
        axs[1, 0].set_title("Weights and Biases")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        plt.tight_layout()
        plt.show()

    def __str__(self):
        layer_str = "\n".join(str(layer) for layer in self.layers)
        return f"MLP:\n{layer_str}"


def main():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]

    mlp = MLP(2, [2, 1], 0.01, True)

    print("-" * NUM_OF_SEPARATORS + "\nPre-training\n" + "-" * NUM_OF_SEPARATORS)
    print(mlp)
    print("-" * NUM_OF_SEPARATORS)

    mlp.train(x, y_train, 200_000)
    print("-" * NUM_OF_SEPARATORS + "\nPost-training\n" + "-" * NUM_OF_SEPARATORS)
    print(mlp)
    print("-" * NUM_OF_SEPARATORS)

    y_pred = mlp.predict(x)
    compare(x, y_train, y_pred)

    mlp.visualize()


if __name__ == "__main__":
    main()
