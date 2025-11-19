import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


L = 20.0         # domain length: x in [0, L)
nx = 400         # number of spatial points
x = np.linspace(0, L, nx, endpoint=False)  # periodic grid
u = 1.0          # advection speed (positive -> right)
CFL = 0.8        # Courant number (must be <= 1 for stability of upwind)
dx = L / nx
dt = CFL * dx / abs(u)  # time step from CFL condition
n_steps = 800    # total number of time steps to animate
plot_stride = 2  # plot every `plot_stride` steps in animation frames

# -------------------------
# INITIAL CONDITION: square pulse
# -------------------------
def g(x, a):
    # square pulse centered near x=10, width=2
    return np.where((x >= (a - 1)) & (x <= (a + 1)), 1.0, 0.0)

def upwind_step(c, u, dt, dx):
    """
    One time-step of the first-order upwind scheme with periodic BCs.
    Uses vectorized roll for neighbors (stable if CFL <= 1).
    """
    lam = u * dt / dx
    if u >= 0:
        # upwind uses left neighbor
        c_left = np.roll(c, 1)
        c_new = c - lam * (c - c_left)
    else:
        # upwind uses right neighbor when velocity negative
        c_right = np.roll(c, -1)
        c_new = c - lam * (c_right - c)
    return c_new

# -------------------------
# exact (analytic) solution for periodic domain
# -------------------------
def exact_solution(x, t, u):
    # shift x by u*t and wrap into [0, L)
    x_shifted = (x - u * t) % L
    return g(x_shifted, 10)

# -------------------------
# ANIMATION SETUP
# -------------------------
c0 = g(x, 10.0)  # initial field

fig, ax = plt.subplots(figsize=(8,4.5))
ax.set_xlim(0, L)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel('x')
ax.set_ylabel('c(x,t)')
ax.set_title(f'Linear advection (upwind), u={u}, CFL={CFL}')

num_line, = ax.plot([], [], lw=2, label='numerical (upwind)')
exact_line, = ax.plot([], [], '--', lw=1.5, label='exact (shift)')
time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)
ax.legend(loc='upper right')

# initialize
c = c0.copy()
t = 0.0

def init():
    num_line.set_data([], [])
    exact_line.set_data([], [])
    time_text.set_text('')
    return num_line, exact_line, time_text

# update function for animation
def update(frame):
    global c, t
    # advance `plot_stride` steps between frames to speed up animation
    for _ in range(plot_stride):
        c = upwind_step(c, u, dt, dx)
        t += dt

    num_line.set_data(x, c)
    exact_line.set_data(x, exact_solution(x, t, u))
    time_text.set_text(f't = {t:.3f}')
    return num_line, exact_line, time_text

ani = FuncAnimation(fig, update, frames=n_steps//plot_stride,
                    init_func=init, blit=True, interval=30)

plt.show()
