import matplotlib.pyplot as plt
import numpy as np


def circular_mic(radius=0.045, center_position=[2.5, 2.5, 1.5], num_mics=6):
    angles = np.linspace(0, 2 * np.pi, num=num_mics, endpoint=False)
    mic_positions = np.zeros((3, num_mics))
    for i, angle in enumerate(angles):
        mic_positions[0, i] = center_position[0] + radius * np.cos(angle)
        mic_positions[1, i] = center_position[1] + radius * np.sin(angle)
        mic_positions[2, i] = center_position[2]  # z-coordinate is constant
    return mic_positions


def plot_microphone_positions(positions: np.ndarray) -> None:
    plt.figure()
    plt.plot(positions[0, :], positions[1, :], "o")
    plt.xlim(2, 3)
    plt.ylim(2, 3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Microphone Array Positions")
    plt.show()
