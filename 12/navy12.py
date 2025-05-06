import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

p = 0.05  # pravdepodobnost spawnu stromu na zemi
f = 0.01  # pravdepodobnost samovzniceni stromu
forest_density = 0.5  # pocatecni hustota lesa
size = 100  # velikost lesa

colors = ["saddlebrown", "green", "orange", "black"]
labels = ["Ground", "Tree", "Fire", "Ash"]
cmap = ListedColormap(colors)

# pocatecni stav lesa
grid = (np.random.rand(size, size) < forest_density).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im = ax1.imshow(grid, cmap=cmap, vmin=0, vmax=3)
ax1.set_title("Forrest Fire")


# aktualizace vizualizace
def update_visuals(grid, t):
    im.set_data(grid)
    # spocitani stavu v lese
    unique, counts = np.unique(grid, return_counts=True)
    count_dict = dict(zip(unique, counts))
    e = count_dict.get(0, 0)
    tr = count_dict.get(1, 0)
    f = count_dict.get(2, 0)
    a = count_dict.get(3, 0)
    # kolacovy graf pro pocty stavu v lese
    ax2.clear()
    values = [e, tr, f, a]
    wedges, _, _ = ax2.pie(values, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    ax2.set_title(f"States in forest (iteration {t})")
    ax2.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.draw()
    plt.pause(0.05)


# kontrola zde jsou v okoli v horici stromy - neumann + moore
def neighborhood_fire(grid, x, y, n_type="neumann"):
    # kotrola pouze vlevo, vpravo, nahore a dole
    if x > 0:
        if grid[x - 1][y] == 2:
            return True
    if x < grid.shape[0] - 1:
        if grid[x + 1][y] == 2:
            return True
    if y > 0:
        if grid[x][y - 1] == 2:
            return True
    if y < grid.shape[1] - 1:
        if grid[x][y + 1] == 2:
            return True
    # pridani uhlopricek
    if n_type == "moore":
        if x > 0 and y > 0:
            if grid[x - 1][y - 1] == 2:
                return True
        if x < grid.shape[0] - 1 and y < grid.shape[1] - 1:
            if grid[x + 1][y + 1] == 2:
                return True
        if x < grid.shape[0] - 1 and y > 0:
            if grid[x + 1][y - 1] == 2:
                return True
        if x > 0 and y < grid.shape[1] - 1:
            if grid[x - 1][y + 1] == 2:
                return True

    return False


# iterace simulace
def iteration(grid):
    new_grid = grid.copy()
    for i in range(size):
        for j in range(size):
            val = grid[i, j]
            if val == 0 and random.random() <= p:  # zem - sance na spawn stromu
                new_grid[i, j] = 1
            elif val == 1 and (
                neighborhood_fire(grid, i, j) or random.random() <= f
            ):  # strom - pozar v okoli nebo sance na samovzniceni
                new_grid[i, j] = 2
            elif val == 2:  # prevod pozaru na popel
                new_grid[i, j] = 3
            elif (
                val == 3
            ):  # prevod popela na zem, zvysena sance na spawn stromu, protoze popel je urodny
                new_grid[i, j] = 1 if random.random() <= p * 5 else 0
    return new_grid


# loop simulace
for t in range(1000):
    grid = iteration(grid)
    update_visuals(grid, t)

plt.show()
