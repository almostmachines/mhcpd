"""
Metropolis-Hastings Change-Point Detection
==============================================
Detecting when a change occurred in time over a 24-hour period.

This demonstrates MCMC on a problem where the evidence p(data)
is analytically intractable.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

rng = np.random.default_rng()

# =============================================================================
# Definitions
# =============================================================================

# Reality
TRUE_TAU = 14.5          # Change happens at 2:30 PM (unknown)
TRUE_MU1 = 12.3         # Mean oxygen level before change (mg/L) (unknown)
TRUE_MU2 = 13.2         # Mean oxygen level after change (mg/L) (unknown)
TRUE_SIGMA = 0.9        # Standard deviation of both regimes (known)
TRUE_EFFECT_SIZE = TRUE_MU2 - TRUE_MU1

# Data
N_OBSERVATIONS = 300
OBSERVATION_TIMES = np.sort(rng.uniform(0, 24, N_OBSERVATIONS))   # NumPy array of timestamps e.g. [0.13, 0.79, 1.24, 1.90, 2,43]
DATA = np.zeros(N_OBSERVATIONS)                                         # NumPy array of zeroes, to become an array of values from two Normal distributions

for i, t in enumerate(OBSERVATION_TIMES):
    if t < TRUE_TAU:
        DATA[i] = rng.normal(TRUE_MU1, TRUE_SIGMA)
    else:
        DATA[i] = rng.normal(TRUE_MU2, TRUE_SIGMA)

# Prior beliefs
PRIOR_MU1_MU2 = 15.0
PRIOR_SIGMA = 5.0

# Initial hypothesis
INITIAL_HYPOTHESIS_TAU = 12.0  # Start at noon
INITIAL_HYPOTHESIS_MU1 = DATA[OBSERVATION_TIMES < INITIAL_HYPOTHESIS_TAU].mean() if np.any(OBSERVATION_TIMES < INITIAL_HYPOTHESIS_TAU) else 15.0
INITIAL_HYPOTHESIS_MU2 = DATA[OBSERVATION_TIMES >= INITIAL_HYPOTHESIS_TAU].mean() if np.any(OBSERVATION_TIMES >= INITIAL_HYPOTHESIS_TAU) else 15.0

# Algorithm settings
TAU_PROPOSAL_WIDTH = 0.1    # Standard deviation for τ proposals (hours)
MU_PROPOSAL_WIDTH = 0.2     # Standard deviation for μ proposals (mg/L)
N_SAMPLES=20000
BURN_IN_ITERATIONS=2000

def print_title(text):
    print("="*60)
    print(text)
    print("="*60)

def hours_to_time(h):
    hours = int(h)
    minutes = int((h - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def log_likelihood(tau, mu1, mu2, data):
    """
    Compute log-likelihood
    
    For any τ, we split observations by their observation times.
    """
    # Boolean mask: which observations are before the change-point?
    before_mask = OBSERVATION_TIMES < tau
    before_data = data[before_mask]

    if len(before_data) > 0:
        ll_before = -0.5 * np.sum(((before_data - mu1) / TRUE_SIGMA) ** 2)
        ll_before -= len(before_data) * np.log(TRUE_SIGMA * np.sqrt(2 * np.pi))
    else:
        ll_before = 0
    
    after_data = data[~before_mask]

    if len(after_data) > 0:
        ll_after = -0.5 * np.sum(((after_data - mu2) / TRUE_SIGMA) ** 2)
        ll_after -= len(after_data) * np.log(TRUE_SIGMA * np.sqrt(2 * np.pi))
    else:
        ll_after = 0
    
    return ll_before + ll_after

def log_prior(tau, mu1, mu2):
    """
    Prior distributions
    """
    if tau < 0 or tau > 24:
        return -np.inf

    # log of uniform density on [0, 24] is log(1/24) = -log(24)
    ll_tau = -np.log(24)
    ll_mu1 = -0.5 * ((mu1 - PRIOR_MU1_MU2) / PRIOR_SIGMA) ** 2
    ll_mu2 = -0.5 * ((mu2 - PRIOR_MU1_MU2) / PRIOR_SIGMA) ** 2
    
    return ll_tau + ll_mu1 + ll_mu2

def log_posterior(tau, mu1, mu2, data):
    """Log-posterior = log-likelihood + log-prior"""
    lp = log_prior(tau, mu1, mu2)

    if np.isinf(lp):
        return -np.inf

    return lp + log_likelihood(tau, mu1, mu2, data)

def metropolis_hastings():
    """
    Metropolis-Hastings
    """

    # Initialize
    tau_current = INITIAL_HYPOTHESIS_TAU
    mu1_current = INITIAL_HYPOTHESIS_MU1
    mu2_current = INITIAL_HYPOTHESIS_MU2
    
    # Samples
    samples_tau = []
    samples_mu1 = []
    samples_mu2 = []
    
    accepted = 0
    current_log_post = log_posterior(tau_current, mu1_current, mu2_current, DATA)
    
    for i in range(N_SAMPLES + BURN_IN_ITERATIONS):
        # Propose new hypothesis
        tau_proposal = tau_current + rng.normal(0, TAU_PROPOSAL_WIDTH)
        mu1_proposal = mu1_current + rng.normal(0, MU_PROPOSAL_WIDTH)
        mu2_proposal = mu2_current + rng.normal(0, MU_PROPOSAL_WIDTH)
        
        # Compute proposed log-posterior
        proposed_log_post = log_posterior(tau_proposal, mu1_proposal, mu2_proposal, DATA)
        
        # Accept/reject
        log_ratio = proposed_log_post - current_log_post
        
        if np.log(rng.random()) < log_ratio:
            tau_current, mu1_current, mu2_current = tau_proposal, mu1_proposal, mu2_proposal
            current_log_post = proposed_log_post

            if i >= BURN_IN_ITERATIONS:
                accepted += 1
        
        # Store sample (after burn-in)
        if i >= BURN_IN_ITERATIONS:
            samples_tau.append(tau_current)
            samples_mu1.append(mu1_current)
            samples_mu2.append(mu2_current)
    
    acceptance_rate = accepted / (N_SAMPLES + BURN_IN_ITERATIONS)
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    
    return np.array(samples_tau), np.array(samples_mu1), np.array(samples_mu2)


def plot_3d_posterior(samples_tau, samples_mu1, samples_mu2):
    """Interactive 3D scatter plot of the joint posterior (τ, μ₁, μ₂)."""
    # Thin samples for performance
    step = max(1, len(samples_tau) // 4000)
    tau = samples_tau[::step]
    mu1 = samples_mu1[::step]
    mu2 = samples_mu2[::step]

    # Colour by log-posterior density (brighter = higher density)
    log_posts = np.array([
        log_posterior(t, m1, m2, DATA) for t, m1, m2 in zip(tau, mu1, mu2)
    ])

    fig = go.Figure(data=[
        go.Scatter3d(
            x=tau, y=mu1, z=mu2,
            mode='markers',
            marker=dict(
                size=2,
                color=log_posts,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='Log-posterior'),
            ),
            name='Posterior samples',
            hovertemplate=(
                'τ = %{x:.2f}h<br>'
                'μ₁ = %{y:.1f} mg/L<br>'
                'μ₂ = %{z:.1f} mg/L<extra></extra>'
            ),
        ),
        # True values
        go.Scatter3d(
            x=[TRUE_TAU], y=[TRUE_MU1], z=[TRUE_MU2],
            mode='markers',
            marker=dict(size=6, color='red', symbol='x'),
            name=f'True values (τ={TRUE_TAU}, μ₁={TRUE_MU1}, μ₂={TRUE_MU2})',
        ),
        # Posterior mean
        go.Scatter3d(
            x=[samples_tau.mean()], y=[samples_mu1.mean()], z=[samples_mu2.mean()],
            mode='markers',
            marker=dict(size=6, color='lime', symbol='diamond'),
            name=f'Posterior mean (τ={samples_tau.mean():.2f}, μ₁={samples_mu1.mean():.1f}, μ₂={samples_mu2.mean():.1f})',
        ),
    ])

    fig.update_layout(
        title='MCMC Posterior Distribution (τ, μ₁, μ₂)',
        scene=dict(
            xaxis_title='Change-point τ (hours)',
            yaxis_title='μ₁ — mean before (mg/L)',
            zaxis_title='μ₂ — mean after (mg/L)',
        ),
        width=900,
        height=700,
        legend=dict(yanchor='top', y=0.95, xanchor='left', x=0.01),
    )

    fig.write_html('output/posterior_3d.html')


def main():
    print_title("STEP 0: Reality")
    print(f"There was a change-point in dissolved oxygen levels (mg/L) in stream water observed during a 24 hour period")
    print()
    print(f"  {TRUE_TAU}h                                       -- the change-point time [unknown]")
    print(f"  Normal({TRUE_MU1} mg/L, {TRUE_SIGMA}² mg/L²)      -- the distribution of oxygen levels before τ [only sigma is known]")
    print(f"  Normal({TRUE_MU2} mg/L, {TRUE_SIGMA}² mg/L²)      -- the distribution of oxygen levels after τ [only sigma is known]")
    print(f"  {TRUE_EFFECT_SIZE:.1f} mg/L                  -- the effect-size (change in mean oxygen levels) [unknown]")
    print(f"""
    We know the true constant standard deviation of {TRUE_SIGMA} mg/L, however we don't know the change-point
    time (14.5h), the mean oxygen level before the change-point ({TRUE_MU1} mg/L) and after the change-point
    ({TRUE_MU2} mg/L), and the effect size {TRUE_EFFECT_SIZE:.1f}. We will estimate those using observed data and a Bayesian model.
    """)
    print()

    print_title("STEP 1: Our observed data")
    print(f"We have {N_OBSERVATIONS} oxygen level measurements over 24 hours.")
    print(f"Observation times range from {OBSERVATION_TIMES[0]:.2f}h to {OBSERVATION_TIMES[-1]:.2f}h.")
    print(f"Oxygen levels range from {DATA.min():.1f}mg/L to {DATA.max():.1f}mg/L.")
    print()
    print()

    print_title("STEP 2: Prior beliefs")
    print("Parameters:")
    print("  τ  ~ Uniform(0, 24)         -- the change-point time, continuous on [0, 24] hours")
    print(f"  μ₁ ~ Normal({PRIOR_MU1_MU2}, {PRIOR_SIGMA}²)   -- mean before change")
    print(f"  μ₂ ~ Normal({PRIOR_MU1_MU2}, {PRIOR_SIGMA}²)   -- mean after change")
    print()
    print()

    print_title("STEP 3: Running Metropolis-Hastings")
    print("Starting MCMC sampler...")

    samples_tau, samples_mu1, samples_mu2 = metropolis_hastings()

    print(f"Generated {len(samples_tau)} samples from the posterior.")
    print()
    print()

    print_title("STEP 4: Results")
    print("POINT ESTIMATES (posterior means):")
    print(f"  Change-point τ:  {samples_tau.mean():.2f}h      (true τ: {TRUE_TAU}h)")
    print(f"  Before mean μ₁:  {samples_mu1.mean():.1f} mg/L    (true μ₁: {TRUE_MU1})")
    print(f"  After mean μ₂:   {samples_mu2.mean():.1f} mg/L    (true μ₂: {TRUE_MU2})")
    effect_size = samples_mu2 - samples_mu1
    print(f"  Effect size: {effect_size.mean():.1f} mg/L         (true effect size: {TRUE_EFFECT_SIZE:.1f} mg/L)")
    print()
    print("95% CREDIBLE INTERVALS:")
    tau_low, tau_high = np.percentile(samples_tau, [2.5, 97.5])
    print(f"  τ:  [{tau_low:.2f}h, {tau_high:.2f}h]")
    print(f"  μ₁: [{np.percentile(samples_mu1, 2.5):.1f}, {np.percentile(samples_mu1, 97.5):.1f}] mg/L")
    print(f"  μ₂: [{np.percentile(samples_mu2, 2.5):.1f}, {np.percentile(samples_mu2, 97.5):.1f}] mg/L")
    print(f"  Effect size: [{np.percentile(effect_size, 2.5):.1f}, {np.percentile(effect_size, 97.5):.1f}]")
    print()
    # Some probability queries
    prob_afternoon = (samples_tau > 12).mean()
    print(f"Probability the change happened in the afternoon (after 12:00): {prob_afternoon:.1%}")
    prob_2_to_4 = ((samples_tau > 14) & (samples_tau < 16)).mean()
    print(f"Probability the change happened between 14:00-16:00: {prob_2_to_4:.1%}")
    print()

    # Visualisation
    _, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Plot 1: Raw data with observation times
    ax = axes[0, 0]
    ax.scatter(OBSERVATION_TIMES, DATA, alpha=0.5, s=20, c='steelblue')
    ax.axvline(TRUE_TAU, color='red', linestyle='--', linewidth=2, label=f'True τ = {TRUE_TAU}h')
    ax.axhline(TRUE_MU1, color='blue', linestyle=':', linewidth=2, xmax=TRUE_TAU/24, label=f'True μ₁ = {TRUE_MU1}')
    ax.axhline(TRUE_MU2, color='purple', linestyle=':', linewidth=2, xmin=TRUE_TAU/24, label=f'True μ₂ = {TRUE_MU2}')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Oxygen level (mg/L)')
    ax.set_title('Observed Data Over 24 Hours')
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['0', '6', '12', '18', '24'])
    ax.legend(fontsize=8)

    # Plot 2: Posterior of τ
    ax = axes[0, 1]
    ax.hist(samples_tau, bins=100, density=True, alpha=0.7, edgecolor='none', color='steelblue')
    ax.axvline(TRUE_TAU, color='red', linestyle='--', linewidth=2, label=f'True τ = {TRUE_TAU}h')
    ax.axvline(samples_tau.mean(), color='green', linestyle='-', linewidth=2, label=f'Estimated = {samples_tau.mean():.2f}h')
    ax.set_xlabel('Change-point τ (hours)')
    ax.set_ylabel('Posterior density')
    ax.set_title('When did the change happen?')
    tau_margin = max(0.5, (np.percentile(samples_tau, 99) - np.percentile(samples_tau, 1)) * 0.3)
    ax.set_xlim(np.percentile(samples_tau, 1) - tau_margin, np.percentile(samples_tau, 99) + tau_margin)
    ax.legend(fontsize=8)

    # Plot 3: Posterior of μ₁
    ax = axes[0, 2]
    ax.hist(samples_mu1, bins=50, density=True, alpha=0.7, edgecolor='none', color='steelblue')
    ax.axvline(TRUE_MU1, color='red', linestyle='--', linewidth=2, label=f'True μ₁ = {TRUE_MU1}')
    ax.axvline(samples_mu1.mean(), color='green', linestyle='-', linewidth=2, label=f'Estimated = {samples_mu1.mean():.1f}')
    ax.set_xlabel('μ₁ (mg/L)')
    ax.set_ylabel('Posterior density')
    ax.set_title('Mean oxygen level before change')
    ax.legend(fontsize=8)

    # Plot 4: Posterior of μ₂
    ax = axes[1, 0]
    ax.hist(samples_mu2, bins=50, density=True, alpha=0.7, edgecolor='none', color='steelblue')
    ax.axvline(TRUE_MU2, color='red', linestyle='--', linewidth=2, label=f'True μ₂ = {TRUE_MU2}')
    ax.axvline(samples_mu2.mean(), color='green', linestyle='-', linewidth=2, label=f'Estimated = {samples_mu2.mean():.1f}')
    ax.set_xlabel('μ₂ (mg/L)')
    ax.set_ylabel('Posterior density')
    ax.set_title('Mean oxygen level after change')
    ax.legend(fontsize=8)

    # Plot 5: Joint posterior of τ and effect size
    ax = axes[1, 1]
    ax.scatter(samples_tau[::10], effect_size[::10], alpha=0.3, s=5, c='steelblue', label='Posterior samples')
    ax.axvline(TRUE_TAU, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'True τ = {TRUE_TAU}h')
    ax.axhline(TRUE_EFFECT_SIZE, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'True effect = {TRUE_EFFECT_SIZE:.0f} mg/L')
    ax.axvline(samples_tau.mean(), color='green', linestyle='-', linewidth=1.5, alpha=0.8, label=f'Est. τ = {samples_tau.mean():.2f}h')
    ax.axhline(effect_size.mean(), color='green', linestyle='-', linewidth=1.5, alpha=0.8, label=f'Est. effect = {effect_size.mean():.1f} mg/L')
    ax.plot(samples_tau.mean(), effect_size.mean(), 'x', color='green', markersize=10, markeredgewidth=2)
    ax.set_xlabel('Change-point τ (hours)')
    ax.set_ylabel('Effect size μ₂ - μ₁ (mg/L)')
    ax.set_title('Joint posterior: When & How Much?')
    tau_sub = samples_tau[::10]
    eff_sub = effect_size[::10]
    tau_margin_j = max(0.5, (np.percentile(tau_sub, 99) - np.percentile(tau_sub, 1)) * 0.3)
    eff_margin_j = max(1.0, (np.percentile(eff_sub, 99) - np.percentile(eff_sub, 1)) * 0.3)
    ax.set_xlim(np.percentile(tau_sub, 1) - tau_margin_j, np.percentile(tau_sub, 99) + tau_margin_j)
    ax.set_ylim(np.percentile(eff_sub, 1) - eff_margin_j, np.percentile(eff_sub, 99) + eff_margin_j)
    ax.legend(fontsize=7, loc='upper left')

    # Plot 6: Trace plot showing exploration
    ax = axes[1, 2]
    ax.plot(samples_tau[:1000], alpha=0.7, linewidth=0.5, color='steelblue')
    ax.axhline(TRUE_TAU, color='red', linestyle='--', linewidth=2, label=f'True τ = {TRUE_TAU}h')
    ax.set_xlabel('MCMC iteration')
    ax.set_ylabel('τ (hours)')
    ax.set_title('Trace plot: MCMC exploring τ')
    trace_data = samples_tau[:1000]
    trace_margin = max(0.5, (trace_data.max() - trace_data.min()) * 0.15)
    ax.set_ylim(trace_data.min() - trace_margin, trace_data.max() + trace_margin)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('output/results.png', dpi=150, bbox_inches='tight')
    print("Visualisation saved!")
    print()
    print()

    print_title("Why we needed MCMC")
    print("""
    To obtain the posterior distribution analytically, we'd need a closed form
    solution for the integral:

      p(data) = ∫∫∫ p(data|τ,μ₁,μ₂) · p(τ) · p(μ₁) · p(μ₂) dτ dμ₁ dμ₂

    The problem: as τ varies continuously, the likelihood has discontinuities
    at each observation timestamp (an observation switches from "before" to 
    "after" as τ crosses it). This breaks the analytical tractability.

    MCMC sidesteps this entirely. We never compute p(data) — we just compare
    parameter settings pairwise, and the normalizing constant cancels out.
    """)

    # 3D interactive visualisation
    plot_3d_posterior(samples_tau, samples_mu1, samples_mu2)

    print("Done! Check 2D plots in output/results.png")
    print("Interactive 3D posterior saved to output/posterior_3d.html")

if __name__ == "__main__":
    main()
