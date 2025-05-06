# import potřebných knihoven
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


# generovani trenovacich dat - seed pro replikovatelnost
np.random.seed(42)
n_samples = 100_000
a_train = np.random.uniform(0, 4, n_samples)
x_train = np.random.uniform(0, 1, n_samples)
y_train = a_train * x_train * (1 - x_train)

# train matrix
X_train = np.stack([a_train, x_train], axis=1)

# keras nn - 2 plne propojene skryte vrstvy
model = Sequential(
    [
        Dense(64, input_dim=2, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(1),
    ]
)

# kompilace s durazem na minimalizaci mean square error
model.compile(optimizer=Adam(0.01), loss="mse")

# trenink
model.fit(X_train, y_train, epochs=15, batch_size=256)

# rozsah hodnot pro bifurcation diagram
a_values = np.linspace(0, 4.0, 1000)

# iterace
n_iterations = 1000
n_transient = 200

# nastaveni pocatecni velmi male hodnoty
x_real = 1e-5 * np.ones_like(a_values)
x_pred = x_real.copy()

a_plot_real, x_plot_real = [], []
a_plot_pred, x_plot_pred = [], []

# iterace s loading barem pro kontrolu ze to jede
for i in tqdm(range(n_iterations)):
    # realny vypocet
    x_real = a_values * x_real * (1 - x_real)

    # predikce pomoci nn
    inputs = np.stack([a_values, x_pred], axis=1)
    x_pred = model.predict(inputs, verbose=0).flatten()

    # ukladani hodnot
    if i >= n_transient:
        a_plot_real.extend(a_values)
        x_plot_real.extend(x_real)
        a_plot_pred.extend(a_values)
        x_plot_pred.extend(x_pred)

# graf pro realna a predikovana data
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
axs[0].plot(a_plot_real, x_plot_real, ",", color="black", alpha=0.3)
axs[0].set_title("Bifurcation diagram")
axs[0].set_xlabel("a")
axs[0].set_ylabel("x")
axs[1].plot(a_plot_pred, x_plot_pred, ",", color="blue", alpha=0.3)
axs[1].set_title("Predicted bifurcatin diagram")
axs[1].set_xlabel("a")

for ax in axs:
    ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
