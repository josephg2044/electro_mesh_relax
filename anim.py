# animation code for fun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def mesh(V_b, mask, rho, target=0.001):
    delta = 1.0
    deltas = []
    V_c = V_b.copy()
    while delta > target:
        step = V_c.copy()
        V_c[1:-1, 1:-1] = (
            step[2:, 1:-1]
            + step[:-2, 1:-1]
            + step[1:-1, 2:]
            + step[1:-1, :-2]
            + rho[1:-1, 1:-1]
        ) / 4.0
        V_c[mask] = V_b[mask]
        delta = np.max(np.abs(V_c - step))
        deltas.append(delta)
    return V_c, deltas


hist = [((0.0, -1.0), (40, 40), (60, 60))]


def boundary(t, shape):
    V_b = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True

    R = 7
    k = 100
    r, c = np.indices(shape)

    (vx, vy), (x1, y1), (x2, y2) = hist[-1]
    dx, dy = x2 - x1, y2 - y1
    theta = np.arctan2(dy, dx)

    norm = dx**2 + dy**2
    G = k / norm if norm else 0

    vx += G * np.cos(theta)
    vy += G * np.sin(theta)
    x1 += vx
    y1 += vy
    x2 -= vx
    y2 -= vy

    hist.append(((vx, vy), (x1, y1), (x2, y2)))

    mask[V_b != 0] = True

    rho = np.zeros(shape)
    rho[(c - x1) ** 2 + (r - y1) ** 2 <= R**2] = 10
    rho[(c - x2) ** 2 + (r - y2) ** 2 <= R**2] = -10

    return V_b, mask, rho


M = 100
shape = (M + 1, M + 1)
x = np.linspace(0, M, M + 1)
y = np.linspace(0, M, M + 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(np.zeros(shape), cmap="inferno", origin="lower", extent=[0, M, 0, M])

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf = ax2.plot_surface(
    X, Y, np.zeros(shape), cmap="inferno", linewidth=0, antialiased=False
)


def update(frame):
    print(frame)
    V_b, mask, rho = boundary(frame, shape)

    V, errors = mesh(V_b, mask, rho, target=0.001)

    im.set_data(V)

    for coll in ax2.collections:
        coll.remove()
    ax2.plot_surface(X, Y, V, cmap="inferno", linewidth=0, antialiased=False)
    im.set_clim(V.min(), V.max())

    return (im,)


anim = FuncAnimation(
    fig, update, frames=np.linspace(0, 154, 155), interval=200, blit=False
)

anim.save("anim.gif", writer="pillow", fps=10)
