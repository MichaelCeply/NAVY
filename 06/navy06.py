import tkinter as tk
from tkinter import ttk, messagebox
import math

# Nastavení Matplotlib pro vložení do Tkinter
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# Funkce pro generování L-systému podle zadaného axiomu a pravidel
def generate_l_system(axiom, rules, iterations):
    result = axiom
    for _ in range(iterations):
        new_result = []
        for c in result:
            new_result.append(rules.get(c, c))
        result = "".join(new_result)
    return result


# Funkce pro interpretaci příkazů L-systému a tvorbu seznamu větví
def draw_l_system(commands, angle_deg, step=5):
    subpaths = []
    current_path = []
    x, y = 0, 0
    angle = 0
    stack = []

    current_path.append((x, y))

    for command in commands:
        if command == "F":
            new_x = x + step * math.cos(math.radians(angle))
            new_y = y + step * math.sin(math.radians(angle))
            current_path.append((new_x, new_y))
            x, y = new_x, new_y
        elif command == "+":
            angle += angle_deg
        elif command == "-":
            angle -= angle_deg
        elif command == "[":
            # Uložíme aktuální stav a začneme novou větev
            stack.append((x, y, angle, current_path))
            current_path = [(x, y)]
        elif command == "]":
            # Uložíme aktuální větev a vrátíme se ke stavu z předchozí větve
            if current_path:
                subpaths.append(current_path)
            if stack:
                x, y, angle, parent_path = stack.pop()
                current_path = parent_path
    if current_path:
        subpaths.append(current_path)
    return subpaths


# Tkinter aplikace s integrovaným Matplotlib grafem
class LSysApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x800")

        # Ovládací panel pro vstupy
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Axiom
        ttk.Label(control_frame, text="Axiom:").grid(row=0, column=0, padx=5, pady=5)
        self.axiom_entry = ttk.Entry(control_frame, width=15)
        self.axiom_entry.grid(row=0, column=1, padx=5, pady=5)
        self.axiom_entry.insert(0, "F")

        # Pravidlo pro symbol F (F kód)
        ttk.Label(control_frame, text="Rule:").grid(row=0, column=2, padx=5, pady=5)
        self.fcode_entry = ttk.Entry(control_frame, width=30)
        self.fcode_entry.grid(row=0, column=3, padx=5, pady=5)
        self.fcode_entry.insert(0, "FF+[+F-F-F]-[-F+F+F]")

        # Počet iterací
        ttk.Label(control_frame, text="Iterations:").grid(
            row=0, column=4, padx=5, pady=5
        )
        self.iterations_entry = ttk.Entry(control_frame, width=10)
        self.iterations_entry.grid(row=0, column=5, padx=5, pady=5)
        self.iterations_entry.insert(0, "3")

        # Úhel ve stupních
        ttk.Label(control_frame, text="Degree (°):").grid(
            row=0, column=6, padx=5, pady=5
        )
        self.angle_entry = ttk.Entry(control_frame, width=10)
        self.angle_entry.grid(row=0, column=7, padx=5, pady=5)
        self.angle_entry.insert(0, "22.5")

        # Tlačítko pro generování a vykreslení L-systému
        self.draw_button = ttk.Button(control_frame, text="Draw", command=self.on_draw)
        self.draw_button.grid(row=0, column=8, padx=10, pady=5)

        # Matplotlib figure a vložené plátno (canvas)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Navigační panel Matplotlibu
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        self.last_drawn_subpaths = None

    def on_draw(self):
        try:
            axiom = self.axiom_entry.get().strip()
            fcode = self.fcode_entry.get().strip()
            iterations = int(self.iterations_entry.get().strip())
            angle = float(self.angle_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Some values are wrong.")
            return

        rules = {"F": fcode}
        l_string = generate_l_system(axiom, rules, iterations)
        subpaths = draw_l_system(l_string, angle, step=5)
        self.last_drawn_subpaths = subpaths
        self.draw_on_figure(subpaths)

    def draw_on_figure(self, subpaths):
        # Vyčistíme aktuální obsah grafu
        self.ax.clear()

        # Spojíme všechny body pro výpočet rozsahů grafu
        points = [pt for path in subpaths for pt in path]
        if not points:
            self.canvas.draw()
            return

        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # Přidáme malou rezervu kolem kresby
        padding_x = 0.05 * (max_x - min_x) if max_x != min_x else 1
        padding_y = 0.05 * (max_y - min_y) if max_y != min_y else 1

        self.ax.set_xlim(min_x - padding_x, max_x + padding_x)
        self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
        self.ax.set_aspect("equal")

        # Vykreslíme každou větev L-systému zvlášť
        for path in subpaths:
            if len(path) < 2:
                continue
            path_xs = [pt[0] for pt in path]
            path_ys = [pt[1] for pt in path]
            self.ax.plot(path_xs, path_ys, color="black", linewidth=1)

        self.canvas.draw()


if __name__ == "__main__":
    app = LSysApp()
    app.mainloop()
