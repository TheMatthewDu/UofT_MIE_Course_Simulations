"""
Simulates and animates the 1D diffusion equation assuming no convective term on the interval [0, L] with Dirichlet
boundary conditions, using a square pulse as the initial condition.

It simulates the solution to the PDE:

dC     d2C
-- = D ---
dt     dx2
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

# Domain setup
L = 20.0
x = np.linspace(0, L, 400)

# Square pulse initial condition within [0, L]
def g(x):
    return np.where((x >= 9) & (x <= 11), 1.0, 0.0)

# Precompute Fourier coefficients
def a_n(n):
    integrand = lambda x_: g(x_) * np.sin(n * np.pi * x_ / L)
    return (2 / L) * quad(integrand, 0, L)[0]

a_values = [a_n(n) for n in range(1, 51)]  # store first 50 terms

# Heat equation solution with Dirichlet BCs
def f(x, t):
    s = np.zeros_like(x)
    for n, an in enumerate(a_values, start=1):
        s += an * np.exp(-((n * np.pi / L) ** 2) * t) * np.sin(n * np.pi * x / L)
    return s

# Plot setup
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.2)
ax.set_title("1D Diffusion")
ax.set_xlabel("$x$")
ax.set_ylabel("$c(x, t)$")
ax.grid(True)
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    time_text.set_text("")
    return line, time_text

def update(frame):
    t = frame / 25.0  # time evolution
    y = f(x, t)
    line.set_data(x, y)
    time_text.set_text(f"t = {t:.2f} secs")
    return line, time_text

ani = FuncAnimation(
    fig, update, frames=200, init_func=init,
    blit=True, interval=50
)

plt.show()
