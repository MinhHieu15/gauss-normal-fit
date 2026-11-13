"""Fit a normal distribution to a histogram by minimising the sum of squared residuals.

Converted from Gauss.ipynb. (mu, sigma) are found with a simple pattern search:
try a +/- step on each parameter, and halve the step when nothing improves.

Usage:
    python gauss_fit.py            # print results, save figures to figures/, show them
    python gauss_fit.py --no-show  # save figures only
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = Path(__file__).parent / "figures"


def gauss(x, mean, sigma):
    """Probability density of N(mean, sigma^2)."""
    return np.exp(-(x - mean) ** 2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def build_data():
    """32 bins of width 2.5 over [130, 210]; the observed densities start at 160."""
    f = np.array([.0125, .005, .009, .0258, .0262, .0507, .053, .051,
                  .0455, .0365, .042, .020, .007, .007, .007, .0005])
    edges = np.arange(130, 210.1, 2.5)
    centre = (edges[:-1] + edges[1:]) / 2

    data = np.zeros(len(centre))
    start = np.searchsorted(edges, 160)
    data[start:start + len(f)] = f
    return centre, data


def ssr(p, centre, data):
    """Sum of squared residuals between the histogram and the Gaussian model."""
    return np.sum((data - gauss(centre, *p)) ** 2)


def fit(centre, data, mean=160, sigma=7, step=4, tol=1e-4):
    """Pattern search over (mean, sigma). Returns the best fit and the search history."""
    best = ssr([mean, sigma], centre, data)
    history = [(mean, sigma, best)]

    while step > tol:
        improved = False
        for dm, ds in [(step, 0), (-step, 0), (0, step), (0, -step)]:
            s = ssr([mean + dm, sigma + ds], centre, data)
            if s < best:
                best, mean, sigma, improved = s, mean + dm, sigma + ds, True
                history.append((mean, sigma, best))
        if not improved:
            step /= 2

    return mean, sigma, best, np.array(history)


def plot_histogram(p, centre, data, title=None, path=None):
    """Overlay the model on the data and return the SSR."""
    mean, sigma = p
    g = gauss(centre, mean, sigma)
    err = np.sum((data - g) ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centre, data, width=2.5, alpha=0.5, label="Data")
    ax.bar(centre, g, width=2.5, alpha=0.5, label="Model")
    xs = np.linspace(centre[0], centre[-1], 400)
    ax.plot(xs, gauss(xs, mean, sigma), color="C1", lw=1.5)
    ax.set_title(title or f"μ = {mean:.2f}, σ = {sigma:.2f}, SSR = {err:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return err


def plot_landscape(centre, data, history, path=None):
    """SSR surface over (mean, sigma) with the search path on top."""
    means = np.linspace(155, 195, 200)
    sigmas = np.linspace(3, 20, 200)
    M, S = np.meshgrid(means, sigmas)
    Z = np.array([[ssr([m, s], centre, data) for m in means] for s in sigmas])

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(M, S, np.log10(Z), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10(SSR)")
    ax.plot(history[:, 0], history[:, 1], "w.-", lw=1, ms=4, label="Search path")
    ax.plot(*history[0, :2], "o", color="orange", label="Start")
    ax.plot(*history[-1, :2], "r*", ms=14, label="Optimum")
    ax.set_xlabel("μ")
    ax.set_ylabel("σ")
    ax.set_title("SSR surface and search path")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)


def plot_convergence(history, path=None):
    """SSR after each accepted step."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history[:, 2], "o-", ms=3)
    ax.set_xlabel("Accepted step")
    ax.set_ylabel("SSR (log scale)")
    ax.set_title("SSR convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="don't open figure windows")
    args = parser.parse_args()

    # Small SSR example on two vectors
    f = np.array([0.0125, 0.005, 0.009])
    g = np.array([0.010, 0.006, 0.012])
    print("f - g      :", f - g)
    print("(f - g)^2  :", (f - g) ** 2)
    print("SSR        :", np.sum((f - g) ** 2))
    print("gauss(160, 160, 7) =", gauss(160, 160, 7))

    centre, data = build_data()
    IMG_DIR.mkdir(exist_ok=True)

    plot_histogram([160, 7], centre, data,
                   title="Initial guess: μ = 160, σ = 7",
                   path=IMG_DIR / "initial_guess.png")

    mean, sigma, best, history = fit(centre, data)
    print(f"\nFit: μ = {mean:.2f}, σ = {sigma:.2f}, SSR = {best:.6f}")
    print(f"Accepted steps: {len(history) - 1}")

    plot_histogram([mean, sigma], centre, data, path=IMG_DIR / "fitted.png")
    plot_landscape(centre, data, history, path=IMG_DIR / "ssr_landscape.png")
    plot_convergence(history, path=IMG_DIR / "convergence.png")
    print(f"Figures saved to {IMG_DIR}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
