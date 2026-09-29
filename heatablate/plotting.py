"""Shared plotting helpers for the validation and convergence scripts.

Forces the non-interactive Agg backend so these run headlessly (CI,
remote sessions, machines with no display) and just write PNG files.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def savefig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved: {os.path.relpath(path)}")
    return path
