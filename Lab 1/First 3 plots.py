import numpy as np
import matplotlib.pyplot as plt

# Euler method simulation function
def simulate_decay_euler(a, b, A0, B0, C0, t_max, dt):
    t = np.arange(0, t_max + dt, dt)
    A = np.zeros_like(t)
    B = np.zeros_like(t)
    C = np.zeros_like(t)

    A[0], B[0], C[0] = A0, B0, C0

    for i in range(1, len(t)):
        dA = -a * A[i-1]
        dB = a * A[i-1] - b * B[i-1]
        dC = b * B[i-1]

        A[i] = A[i-1] + dA * dt
        B[i] = B[i-1] + dB * dt
        C[i] = C[i-1] + dC * dt

    return t, A, B, C

# Simulation parameters
A0, B0, C0 = 1.0, 0.0, 0.0
t_max = 50
dt = 0.01

# Cases to simulate and plot
cases = [
    (0.1, 1, "decay_case1.png", "a = 0.1, b = 1"),
    (1, 0.1, "decay_case2.png", "a = 1, b = 0.1"),
    (1, 1, "decay_case3.png", "a = 1, b = 1")
]

# Loop through cases
for a, b, filename, title in cases:
    t, A, B, C = simulate_decay_euler(a, b, A0, B0, C0, t_max, dt)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, A, label='A(t)', color='blue')
    plt.plot(t, B, label='B(t)', color='orange')
    plt.plot(t, C, label='C(t)', color='green')
    plt.title(f"Euler's Method Simulation: {title}")
    plt.xlabel("Time")
    plt.ylabel("Quantity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

print("Plots saved as decay_case1.png, decay_case2.png, and decay_case3.png")
