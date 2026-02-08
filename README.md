# Metropolis-Hastings Change-Point Detection

Bayesian change-point detection using the Metropolis-Hastings MCMC algorithm. Given noisy time-series observations, the script infers *when* a change occurred and *what* the parameters were before and after the change.

## The Problem

Imagine server response times are recorded over a 24-hour period. At some unknown point, the mean response time shifts (e.g. due to a deployment or infrastructure change). We observe noisy measurements but don't know:

- **When** the change happened (the change-point time, $\tau$)
- **What** the mean response time was before ($\mu_1$) and after ($\mu_2$) the change

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

Results are printed to the terminal and a visualization is saved to `output/mcmc_continuous_results.png`.

## Dependencies

- [NumPy](https://numpy.org/) -- numerical computation
- [Matplotlib](https://matplotlib.org/) -- plotting
