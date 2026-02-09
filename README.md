# Metropolis-Hastings Change-Point Detection

Bayesian change-point detection using the Metropolis-Hastings MCMC algorithm. Given noisy time-series observations, the script infers *when* a change occurred and *what* the parameters were before and after the change.

## The Problem

Imagine that the levels of dissolved oxygen (mg/L) in stream water are sampled at random over a 24 hour period. At some unknown time, the mean oxygen level shifts. We observe noisy measurements but don't know:

- **When** the change happened (the change-point time, $\tau$)
- **What** the mean oxygen level was before ($\mu_1$) and after ($\mu_2$) the change

Computing the posterior distribution analytically would require integrating over a likelihood with discontinuities at every observation timestamp, making it intractable. Metropolis-Hastings sidesteps this by sampling from the posterior directly, never needing to compute the normalizing constant $p(\text{data})$.

## Installation

Requires Python and [uv](https://docs.astral.sh/uv/).

```sh
git clone <repo-url>
cd mhcpd
uv sync
```

## Usage

```sh
uv run main.py
```

Results are printed to the terminal and 2D plots are saved to `output/results.png`.

An interactive 3D visualisation of the joint posterior distribution ($\tau$, $\mu_1$, $\mu_2$) is saved to `output/posterior_3d.html`. Open it in a browser to pan, zoom, and rotate.

![3D posterior distribution](./plotly-screenshot.png)

## Dependencies

- [NumPy](https://numpy.org/) -- numerical computation
- [Matplotlib](https://matplotlib.org/) -- plotting
- [Plotly](https://plotly.com/python/) -- interactive 3D visualisation

## License

[MIT](LICENSE)
