# ======================================================
# BEC rotation curves: Rotating vs Non-rotating BEC
# Multiple galaxies loop
# ======================================================

import numpy as np
import arviz as az
import corner
import numpyro
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
from functools import partial

jax.config.update("jax_enable_x64", True)

# ----------------------------------------
# BEC halo velocity
# ----------------------------------------
def jax_vhalo_bec_full(params, r):
    rho_c = 10.0 ** params['log_rho_c']
    R_bec = params['R_bec']
    Omega  = params['Omega']
    factor = 80.861 * (rho_c / 1e-24) * (R_bec ** 2)
    x = jnp.pi * r / R_bec
    term1 = (1.0 - Omega**2) * jnp.sin(x) / x
    term2 = -(1.0 - Omega**2) * jnp.cos(x)
    term3 = (Omega**2 / 3.0) * x**2
    halo_v2 = factor * (term1 + term2 + term3)
    return jnp.sqrt(jnp.clip(halo_v2, a_min=0.0))

def jax_vmod_bec(params, r):
    return jnp.sqrt(
        jax_vhalo_bec_full(params, r)**2 +
        jnp.interp(
            r, params['r'],
            params['vg']**2
            + 10.0**params['log_mld'] * params['vd']**2
            + 10.0**params['log_mlb'] * params['vb']**2
        )
    )

# ----------------------------------------
# Models
# ----------------------------------------
def model_bec_rot(t, y_err, y, params):
    log_rho_c = numpyro.sample("log_rho_c", dist.Uniform(-26, -22))
    R_bec     = numpyro.sample("R_bec",    dist.Uniform(0.0, 30.0))
    Omega     = numpyro.sample("Omega",    dist.Uniform(0.0, 1.0))
    log_mld   = numpyro.sample("log_mld",  dist.Normal(-0.3, 0.1))
    log_mlb   = numpyro.sample("log_mlb",  dist.Normal(-0.15, 0.1))
    p = {'log_rho_c': log_rho_c, 'R_bec': R_bec, 'Omega': Omega,
         'log_mld': log_mld, 'log_mlb': log_mlb, **params}
    v_model = jax_vmod_bec(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

def model_bec_norot(t, y_err, y, params):
    log_rho_c = numpyro.sample("log_rho_c", dist.Uniform(-26, -22))
    R_bec     = numpyro.sample("R_bec",    dist.Uniform(0.0, 30.0))
    log_mld   = numpyro.sample("log_mld",  dist.Normal(-0.3, 0.1))
    log_mlb   = numpyro.sample("log_mlb",  dist.Normal(-0.15, 0.1))
    p = {'log_rho_c': log_rho_c, 'R_bec': R_bec, 'Omega': 0.0,
         'log_mld': log_mld, 'log_mlb': log_mlb, **params}
    v_model = jax_vmod_bec(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

# ----------------------------------------
# Evidence helpers (same as before)
# ----------------------------------------
def _loglike_normal(y, mu, sigma):
    return -0.5 * jnp.sum(((y - mu) / sigma)**2 + 2.0*jnp.log(sigma) + jnp.log(2*jnp.pi))

@partial(jax.jit, static_argnames=("is_rotating",))
def _batch_loglike_bec(samples, is_rotating, r_arr, y_arr, yerr_arr, data_params):
    log_rho_c = samples['log_rho_c']
    R_bec     = samples['R_bec']
    log_mld   = samples['log_mld']
    log_mlb   = samples['log_mlb']
    if is_rotating:
        Omega = samples['Omega']
    def _one_draw(i):
        p = {'log_rho_c': log_rho_c[i], 'R_bec': R_bec[i],
             'log_mld': log_mld[i], 'log_mlb': log_mlb[i],
             'Omega': (Omega[i] if is_rotating else 0.0), **data_params}
        v = jax_vmod_bec(p, r_arr)
        return _loglike_normal(y_arr, v, yerr_arr)
    return jax.vmap(_one_draw)(jnp.arange(log_rho_c.shape[0]))

def _harmonic_lnZ_from_loglikes(loglikes_np):
    b = -loglikes_np
    b_min = b.min()
    mean_term = np.mean(np.exp(b - b_min))
    return -(b_min + np.log(mean_term))

def _bic_from_loglikes(loglikes_np, k, n):
    ll_hat = float(loglikes_np.max())
    bic = k*np.log(n) - 2.0*ll_hat
    return bic, ll_hat

def _prep_samples(d, keys, max_draws=8000):
    out = {k: np.asarray(d[k]).reshape(-1) for k in keys}
    nd = len(out[keys[0]])
    if nd > max_draws:
        sel = np.random.choice(nd, size=max_draws, replace=False)
        for k in keys: out[k] = out[k][sel]
    return {k: jnp.array(v) for k,v in out.items()}

# ----------------------------------------
# Main loop: multiple galaxies
# ----------------------------------------
galaxies = {
# Choose the galaxies you want to run the model on ___________________________________________________________________________
    
}

num_warmup  = 1500
num_samples = 4000
num_chains  = 2
accept_prob = 0.9

for gal_name, filepath in galaxies.items():
    print(f"\n===== {gal_name} =====")

    # load data
    r, vobs, e_vobs, vg, vd, vb, _, _ = np.genfromtxt(filepath, unpack=True)
    data_params = {'r': r, 'vg': vg, 'vd': vd, 'vb': vb}
    n_data = len(r)

    # Run rotating
    sampler_rot = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_bec_rot, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_rot.run(jax.random.PRNGKey(0), r, e_vobs, vobs, data_params)
    posterior_rot = sampler_rot.get_samples()

    # Run non-rotating
    sampler_norot = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_bec_norot, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_norot.run(jax.random.PRNGKey(1), r, e_vobs, vobs, data_params)
    posterior_norot = sampler_norot.get_samples()

   # =============================
    # Combined corner plot per galaxy (with Ω row/col)
    # =============================
    param_order = ["log_rho_c", "R_bec", "log_mld", "log_mlb", "Omega"]
    labels = [
        r"$\log \rho_c\ \mathrm{(g\ cm^{-3})}$",
        r"$R_{\rm BEC}\ \mathrm{(kpc)}$",
        r"$\log (M/L)_d$",
        r"$\log (M/L)_b$",
        r"$\Omega$"
    ]

    # Rotating BEC: real Ω samples
    X_rot = np.column_stack([np.asarray(posterior_rot[p]) for p in param_order])

    # Non-rotating BEC: pad Ω with zeros so we can show the 5th row/col
    omega_dummy = np.zeros(len(posterior_norot["log_rho_c"]))
    X_norot = np.column_stack([
        np.asarray(posterior_norot["log_rho_c"]),
        np.asarray(posterior_norot["R_bec"]),
        np.asarray(posterior_norot["log_mld"]),
        np.asarray(posterior_norot["log_mlb"]),
        omega_dummy,
    ])

    # Optional thinning
    def _thin(X, n_max=4000):
        if X.shape[0] <= n_max: 
            return X
        idx = np.random.choice(X.shape[0], size=n_max, replace=False)
        return X[idx]

    X_rot   = _thin(X_rot,   4000)
    X_norot = _thin(X_norot, 4000)

    # Shared axis ranges (include Ω too)
    ranges = []
    for i in range(X_rot.shape[1]):
        both = np.concatenate([X_rot[:, i], X_norot[:, i]])
        lo, hi = np.percentile(both, [0.5, 99.5])
        pad = 0.03 * (hi - lo) if hi > lo else 1e-3
        ranges.append((lo - pad, hi + pad))

    # Plot: Rotating (filled red), Non-rotating (blue dashed; Ω=0 line)
    fig = corner.corner(
        X_rot,
        labels=labels,
        color="tab:red",
        bins=40,
        smooth=1.2,
        range=ranges,
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.393, 0.865],  # ~1σ, 2σ
        title_kwargs={"fontsize": 10},
    )

    corner.corner(
        X_norot,
        fig=fig,
        color="tab:blue",
        bins=40,
        smooth=1.2,
        range=ranges,
        plot_datapoints=False,
        fill_contours=False,
        levels=[0.393, 0.865],
        contour_kwargs={"linestyles": "--", "linewidths": 1.4},
    )

    # Legend & save
    handles = [
        mlines.Line2D([], [], color="tab:red",  lw=5, label="Rotating BEC ($\\Omega$ free)"),
        mlines.Line2D([], [], color="tab:blue", lw=2, ls="--", label="Non-rotating BEC ($\\Omega=0$)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=11)

    plt.suptitle(f"{gal_name}: Rotating vs Non-rotating BEC (with $\\Omega$ axis)", y=0.99, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"corner_bec_with_omega_{gal_name}.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


    # =============================
    # Evidence calcs (per galaxy)
    # =============================
    keys_norot = ["log_rho_c", "R_bec", "log_mld", "log_mlb"]
    s_norot = _prep_samples(posterior_norot, keys_norot)
    loglikes_norot = np.asarray(_batch_loglike_bec(
        s_norot, False, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params))
    bic_norot, llhat_norot = _bic_from_loglikes(loglikes_norot, k=4, n=n_data)
    lnZ_harm_norot = _harmonic_lnZ_from_loglikes(loglikes_norot)
    lnZ_bic_norot  = -0.5*bic_norot

    keys_rot = ["log_rho_c", "R_bec", "log_mld", "log_mlb", "Omega"]
    s_rot = _prep_samples(posterior_rot, keys_rot)
    loglikes_rot = np.asarray(_batch_loglike_bec(
        s_rot, True, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params))
    bic_rot, llhat_rot = _bic_from_loglikes(loglikes_rot, k=5, n=n_data)
    lnZ_harm_rot = _harmonic_lnZ_from_loglikes(loglikes_rot)
    lnZ_bic_rot  = -0.5*bic_rot

    print(f"Non-rot BEC: max logL={llhat_norot:.3f}, BIC={bic_norot:.3f}, lnZ_harm={lnZ_harm_norot:.3f}, lnZ_BIC={lnZ_bic_norot:.3f}")
    print(f"Rot   BEC:   max logL={llhat_rot:.3f},   BIC={bic_rot:.3f}, lnZ_harm={lnZ_harm_rot:.3f}, lnZ_BIC={lnZ_bic_rot:.3f}")
    print(f"ΔBIC={bic_rot-bic_norot:.3f}, ΔlnZ_harm={lnZ_harm_rot-lnZ_harm_norot:.3f}, ΔlnZ_BIC={lnZ_bic_rot-lnZ_bic_norot:.3f}")
