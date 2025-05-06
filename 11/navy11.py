import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# pocatecni parametry kyvadel - delky, hmotnosti, uhly a uhlove rychlosti
g = 10
l1 = np.random.rand() * 2
l2 = 2 - l1
m1 = np.random.rand()
m2 = np.random.rand()

theta1 = np.pi * np.random.rand()
theta2 = np.pi * np.random.rand()
omega1 = 0.0
omega2 = 0.0

print(f"l1: {l1}, l2: {l2}\nm1: {m1}, m2: {m2}\ntheta1: {theta1}, theta2: {theta2}")

state0 = [theta1, omega1, theta2, omega2]

# timestamps - kazdych 10ms po dobu 20s
t = np.linspace(0, 20, 2000)


# derivace
def get_derivatives(state, t, l1, l2, m1, m2):
    th1, w1, th2, w2 = state

    delta = th2 - th1
    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    denom2 = (l2 / l1) * denom1

    dw1 = (
        m2 * l1 * w1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(th2) * np.cos(delta)
        + m2 * l2 * w2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(th1)
    ) / denom1

    dw2 = (
        -m2 * l2 * w2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
        - (m1 + m2) * l1 * w1**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(th2)
    ) / denom2

    return [w1, dw1, w2, dw2]


# rovnice pohybu
sol = odeint(get_derivatives, state0, t, args=(l1, l2, m1, m2))

# pozice pro animace
theta1 = sol[:, 0]
theta2 = sol[:, 2]

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# vykreslovaci smycka
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.grid()

(line,) = ax.plot([], [], "o-", lw=2, color="darkred")
(trace,) = ax.plot([], [], "-", lw=1, alpha=0.5, color="blue")
trail_x, trail_y = [], []


def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace


def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)

    trail_x.append(x2[i])
    trail_y.append(y2[i])
    if len(trail_x) > 100:
        trail_x.pop(0)
        trail_y.pop(0)
    trace.set_data(trail_x, trail_y)
    return line, trace


ani = animation.FuncAnimation(
    fig, update, frames=len(t), init_func=init, interval=10, blit=True
)

plt.title("Double Pendulum")
plt.show()
