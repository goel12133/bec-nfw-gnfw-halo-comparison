import numpy as np
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

# ------------------------
# Constants
# ------------------------
G_KPC = 4.30091e-6
H0    = 70.0
Dc    = 200.0

def jax_rho_crit():
    H_kpc = H0 / 1000.0
    return 3.0 * H_kpc**2 / (8.0 * jnp.pi * G_KPC)

def jax_Rvir(Mh):
    rho_c = jax_rho_crit()
    return (Mh / ((4.0/3.0) * jnp.pi * Dc * rho_c))**(1.0/3.0)

def jax_Vvir(Mh):
    Rv = jax_Rvir(Mh)
    return jnp.sqrt(G_KPC * Mh / Rv)

def jax_fc(x):
    return jnp.log1p(x) - x / (1.0 + x)

# ------------------------
# NFW halo + total model
# ------------------------
def jax_vhalo_nfw(params, R):
    Mh = 10.0 ** params['log_mh']
    cc = 10.0 ** params['log_c']
    rv = jax_Rvir(Mh)
    R_safe = jnp.clip(R, 1e-3, None)
    numer = jax_fc(cc * R_safe / rv)
    denom = jnp.clip(jax_fc(cc), 1e-6, None)
    vel2 = jax_Vvir(Mh)**2 * rv / R_safe * numer / denom
    return jnp.sqrt(jnp.clip(vel2, 0.))

def jax_vmod_nfw(params, r_eval):
    vhalo = jax_vhalo_nfw(params, r_eval)
    bary_sq = jnp.interp(r_eval, params['r'],
        params['vg']**2 +
        10.0**params['log_mld']*params['vd']**2 +
        10.0**params['log_mlb']*params['vb']**2
    )
    return jnp.sqrt(vhalo**2 + bary_sq)

def model_nfw(t, y_err, y, params):
    log_mh  = numpyro.sample("log_mh",  dist.Uniform(8., 13.))
    log_c   = numpyro.sample("log_c",   dist.Uniform(0.3, 1.0))
    log_mld = numpyro.sample("log_mld", dist.Normal(-0.3, 0.1))
    log_mlb = numpyro.sample("log_mlb", dist.Normal(-0.15, 0.1))
    p = {'log_mh': log_mh, 'log_c': log_c,
         'log_mld': log_mld, 'log_mlb': log_mlb, **params}
    v_model = jax_vmod_nfw(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

# ------------------------
# gNFW
# ------------------------
_gl_x, _gl_w = np.polynomial.legendre.leggauss(64)
GL_X, GL_W = jnp.array(_gl_x), jnp.array(_gl_w)

def _gnfw_integrand(t, alpha):
    return t**(2.0-alpha) * (1.0+t)**(alpha-3.0)

def jax_gnfw_f(x, alpha):
    x = jnp.atleast_1d(x)
    def _quad_single(xi):
        t = 0.5*xi*(GL_X+1.0)
        return 0.5*xi*jnp.sum(GL_W * _gnfw_integrand(jnp.clip(t,1e-12,None), alpha))
    return jax.vmap(_quad_single)(x)

def jax_vhalo_gnfw(params, R):
    Mh, cc, alpha = 10.0**params['log_mh'], 10.0**params['log_c'], params['alpha']
    R_safe = jnp.clip(R, 1e-6, None)
    Rvir = jax_Rvir(Mh)
    rs = Rvir/cc
    x, c = R_safe/rs, cc
    fx, fc = jax_gnfw_f(x, alpha), jax_gnfw_f(c, alpha)[0]
    rho_s = Mh/(4.0*jnp.pi*rs**3*jnp.clip(fc,1e-20,None))
    Mr = 4.0*jnp.pi*rho_s*rs**3*fx
    return jnp.sqrt(jnp.clip(G_KPC*Mr/R_safe, 0.))

def jax_vmod_gnfw(params, r_eval):
    vhalo = jax_vhalo_gnfw(params, r_eval)
    vg2_signed = jnp.sign(params['vg'])*params['vg']**2
    bary_sq_profile = (
        10.0**params['log_mld']*params['vd']**2 +
        10.0**params['log_mlb']*params['vb']**2 +
        vg2_signed
    )
    bary_sq = jnp.interp(r_eval, params['r'], bary_sq_profile)
    return jnp.sqrt(jnp.clip(vhalo**2 + bary_sq, 0.))

def model_gnfw(t, y_err, y, params):
    log_mh  = numpyro.sample("log_mh",  dist.Uniform(8., 13.))
    log_c   = numpyro.sample("log_c",   dist.Uniform(0.3, 1.0))
    alpha   = numpyro.sample("alpha",   dist.Uniform(0.0, 1.5))
    log_mld = numpyro.sample("log_mld", dist.Normal(-0.3, 0.1))
    log_mlb = numpyro.sample("log_mlb", dist.Normal(-0.15, 0.1))
    p = {'log_mh': log_mh, 'log_c': log_c, 'alpha': alpha,
         'log_mld': log_mld, 'log_mlb': log_mlb, **params}
    v_model = jax_vmod_gnfw(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

# ------------------------
# BEC halo + total model
# ------------------------
def jax_vhalo_bec_full(params, r):
    """
    BEC halo circular speed (km/s) with rotation parameter Omega in [0,1].
    params: {'log_rho_c', 'R_bec', 'Omega'}
    r: radii (kpc)
    """
    rho_c = 10.0 ** params['log_rho_c']   # central density (g/cm^3)
    R_bec = params['R_bec']               # BEC radius (kpc)
    Omega  = params['Omega']              # dimensionless spin

    # Prefactor collects constants (unit-consistent to yield km/s downstream)
    factor = 80.861 * (rho_c / 1e-24) * (R_bec ** 2)

    x = jnp.pi * r / R_bec
    term1 = (1.0 - Omega**2) * jnp.sin(x) / x
    term2 = -(1.0 - Omega**2) * jnp.cos(x)
    term3 = (Omega**2 / 3.0) * x**2

    v2 = factor * (term1 + term2 + term3)
    return jnp.sqrt(jnp.clip(v2, a_min=0.0))

def jax_vmod_bec(params, r_eval):
    """Total model speed for BEC = halo ⊕ baryons."""
    vhalo = jax_vhalo_bec_full(params, r_eval)
    bary_sq = jnp.interp(
        r_eval,
        params['r'],
        params['vg']**2
        + 10.0**params['log_mld'] * params['vd']**2
        + 10.0**params['log_mlb'] * params['vb']**2
    )
    return jnp.sqrt(jnp.clip(vhalo**2 + bary_sq, 0.0))

# ------------------------
# BEC models (rotating vs. non-rotating)
# ------------------------
def model_bec_rot(t, y_err, y, params):
    log_rho_c = numpyro.sample("log_rho_c", dist.Uniform(-26.0, -22.0))
    R_bec     = numpyro.sample("R_bec",    dist.Uniform(0.0, 30.0))
    Omega     = numpyro.sample("Omega",    dist.Uniform(0.0, 1.0))
    log_mld   = numpyro.sample("log_mld",  dist.Normal(-0.3, 0.1))
    log_mlb   = numpyro.sample("log_mlb",  dist.Normal(-0.15, 0.1))

    p = {
        'log_rho_c': log_rho_c,
        'R_bec':     R_bec,
        'Omega':     Omega,
        'log_mld':   log_mld,
        'log_mlb':   log_mlb,
        **params
    }
    v_model = jax_vmod_bec(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

def model_bec_norot(t, y_err, y, params):
    log_rho_c = numpyro.sample("log_rho_c", dist.Uniform(-26.0, -22.0))
    R_bec     = numpyro.sample("R_bec",    dist.Uniform(0.0, 30.0))
    log_mld   = numpyro.sample("log_mld",  dist.Normal(-0.3, 0.1))
    log_mlb   = numpyro.sample("log_mlb",  dist.Normal(-0.15, 0.1))

    p = {
        'log_rho_c': log_rho_c,
        'R_bec':     R_bec,
        'Omega':     0.0,    # fixed for non-rotating
        'log_mld':   log_mld,
        'log_mlb':   log_mlb,
        **params
    }
    v_model = jax_vmod_bec(p, t)
    numpyro.sample("obs", dist.Normal(v_model, y_err), obs=y)

# ------------------------
# Batched log-like for BEC (for evidence/BIC)
# ------------------------
@partial(jax.jit, static_argnames=("is_rotating",))
def _batch_loglike_bec(samples, is_rotating, r_arr, y_arr, yerr_arr, data_params):
    """
    Vectorized Normal log-likelihood over BEC posterior draws.

    samples (JAX arrays):
      non-rot: 'log_rho_c','R_bec','log_mld','log_mlb'
      rot    : above + 'Omega'
    """
    log_rho_c = samples['log_rho_c']
    R_bec     = samples['R_bec']
    log_mld   = samples['log_mld']
    log_mlb   = samples['log_mlb']
    Omega     = samples['Omega'] if is_rotating else None

    two_pi = 2.0 * jnp.pi

    def _one_draw(i):
        p = {
            'log_rho_c': log_rho_c[i],
            'R_bec':     R_bec[i],
            'log_mld':   log_mld[i],
            'log_mlb':   log_mlb[i],
            'Omega':     (Omega[i] if is_rotating else 0.0),
            **data_params
        }
        mu = jax_vmod_bec(p, r_arr)
        return -0.5 * jnp.sum(((y_arr - mu) / yerr_arr)**2 + 2.0*jnp.log(yerr_arr) + jnp.log(two_pi))

    n = log_rho_c.shape[0]
    return jax.vmap(_one_draw)(jnp.arange(n))


# ------------------------
# Evidence helpers
# ------------------------
def _loglike_normal(y, mu, sigma):
    return -0.5*jnp.sum(((y-mu)/sigma)**2 + 2.0*jnp.log(sigma) + jnp.log(2*jnp.pi))

@partial(jax.jit, static_argnames=("is_gnfw",))
def _batch_loglike(samples, is_gnfw, r_arr, y_arr, yerr_arr, data_params):
    """
    Vectorized log-likelihood over posterior draws.

    samples: dict of JAX arrays with keys:
      NFW : 'log_c', 'log_mh', 'log_mld', 'log_mlb'
      gNFW: above + 'alpha'
    """
    log_c   = samples['log_c']
    log_mh  = samples['log_mh']
    log_mld = samples['log_mld']
    log_mlb = samples['log_mlb']
    alpha   = samples['alpha'] if is_gnfw else None

    def _one_draw(i):
        p = {
            'log_c':  log_c[i],
            'log_mh': log_mh[i],
            'log_mld': log_mld[i],
            'log_mlb': log_mlb[i],
            **data_params
        }
        if is_gnfw:
            p['alpha'] = alpha[i]  # add alpha BEFORE calling the model
            v = jax_vmod_gnfw(p, r_arr)
        else:
            v = jax_vmod_nfw(p, r_arr)
        # Gaussian log-like summed over data
        two_pi = 2.0 * jnp.pi
        return -0.5 * jnp.sum(((y_arr - v) / yerr_arr)**2 + 2.0*jnp.log(yerr_arr) + jnp.log(two_pi))

    n = log_c.shape[0]
    return jax.vmap(_one_draw)(jnp.arange(n))


def _harmonic_lnZ_from_loglikes(loglikes_np):
    b=-loglikes_np; bmin=b.min()
    return -(bmin+np.log(np.mean(np.exp(b-bmin))))

def _bic_from_loglikes(loglikes_np,k,n):
    ll_hat=float(loglikes_np.max()); bic=k*np.log(n)-2.0*ll_hat
    return bic,ll_hat

def _prep_samples(d,keys,max_draws=8000):
    out={k:np.asarray(d[k]).reshape(-1) for k in keys}
    nd=len(out[keys[0]])
    if nd>max_draws:
        sel=np.random.choice(nd,size=max_draws,replace=False)
        for k in keys: out[k]=out[k][sel]
    return {k:jnp.array(v) for k,v in out.items()}

# ------------------------
# Main loop: galaxies
# ------------------------
galaxies = { 
   # Choose the galaxies you want to run the model on ___________________________________________________________________________
}



# ======================================================
# Run models, collect evidences, and make ΔlnZ histograms
# ======================================================

num_warmup  = 1000
num_samples = 3000
num_chains  = 2
accept_prob = 0.9

# Per-model ΔlnZ (relative to gNFW) across ALL galaxies
delta_harm_NFW     = []
delta_harm_BEC     = []
delta_harm_BEC_rot = []

delta_bic_NFW      = []
delta_bic_BEC      = []
delta_bic_BEC_rot  = []

for gi, (gal_name, filepath) in enumerate(galaxies.items()):
    print(f"\n===== {gal_name} =====")
    r, vobs, e_vobs, vg, vd, vb, _, _ = np.genfromtxt(filepath, unpack=True)
    data_params = {'r': r, 'vg': vg, 'vd': vd, 'vb': vb}
    n_data = len(r)

    # -------------------------
    # gNFW (reference)
    # -------------------------
    sampler_g = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_gnfw, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_g.run(jax.random.PRNGKey(10_000 + 4*gi + 0), r, e_vobs, vobs, data_params)
    posterior_g = sampler_g.get_samples()

    keys_g = ["log_c","log_mh","log_mld","log_mlb","alpha"]
    s_g = _prep_samples(posterior_g, keys_g)
    loglikes_g = np.asarray(
        _batch_loglike(s_g, True, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params)
    )
    bic_g, llhat_g   = _bic_from_loglikes(loglikes_g, k=5, n=n_data)
    lnZ_harm_g       = _harmonic_lnZ_from_loglikes(loglikes_g)
    lnZ_bic_g        = -0.5 * bic_g

    # -------------------------
    # NFW
    # -------------------------
    sampler_n = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_nfw, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_n.run(jax.random.PRNGKey(10_000 + 4*gi + 1), r, e_vobs, vobs, data_params)
    posterior_n = sampler_n.get_samples()

    keys_n = ["log_c","log_mh","log_mld","log_mlb"]
    s_n = _prep_samples(posterior_n, keys_n)
    loglikes_n = np.asarray(
        _batch_loglike(s_n, False, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params)
    )
    bic_n, _        = _bic_from_loglikes(loglikes_n, k=4, n=n_data)
    lnZ_harm_n      = _harmonic_lnZ_from_loglikes(loglikes_n)
    lnZ_bic_n       = -0.5 * bic_n

    # -------------------------
    # BEC (non-rotating)
    # -------------------------
    sampler_b0 = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_bec_norot, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_b0.run(jax.random.PRNGKey(10_000 + 4*gi + 2), r, e_vobs, vobs, data_params)
    posterior_b0 = sampler_b0.get_samples()

    keys_b0 = ["log_rho_c","R_bec","log_mld","log_mlb"]
    s_b0 = _prep_samples(posterior_b0, keys_b0)
    loglikes_b0 = np.asarray(
        _batch_loglike_bec(s_b0, False, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params)
    )
    bic_b0, _      = _bic_from_loglikes(loglikes_b0, k=4, n=n_data)
    lnZ_harm_b0    = _harmonic_lnZ_from_loglikes(loglikes_b0)
    lnZ_bic_b0     = -0.5 * bic_b0

    # -------------------------
    # BEC (rotating)
    # -------------------------
    sampler_b1 = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_bec_rot, dense_mass=True, target_accept_prob=accept_prob),
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False,
    )
    sampler_b1.run(jax.random.PRNGKey(10_000 + 4*gi + 3), r, e_vobs, vobs, data_params)
    posterior_b1 = sampler_b1.get_samples()

    keys_b1 = ["log_rho_c","R_bec","log_mld","log_mlb","Omega"]
    s_b1 = _prep_samples(posterior_b1, keys_b1)
    loglikes_b1 = np.asarray(
        _batch_loglike_bec(s_b1, True, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params)
    )
    bic_b1, _      = _bic_from_loglikes(loglikes_b1, k=5, n=n_data)
    lnZ_harm_b1    = _harmonic_lnZ_from_loglikes(loglikes_b1)
    lnZ_bic_b1     = -0.5 * bic_b1

    # -------------------------
    # Store ΔlnZ relative to gNFW
    # -------------------------
    delta_harm_NFW.append(     lnZ_harm_n  - lnZ_harm_g)
    delta_harm_BEC.append(     lnZ_harm_b0 - lnZ_harm_g)
    delta_harm_BEC_rot.append( lnZ_harm_b1 - lnZ_harm_g)

    delta_bic_NFW.append(      lnZ_bic_n   - lnZ_bic_g)
    delta_bic_BEC.append(      lnZ_bic_b0  - lnZ_bic_g)
    delta_bic_BEC_rot.append(  lnZ_bic_b1  - lnZ_bic_g)

    # quick print for sanity
    print(f"ΔlnZ_harm: NFW={delta_harm_NFW[-1]:+.3f}, BEC={delta_harm_BEC[-1]:+.3f}, BEC(rot)={delta_harm_BEC_rot[-1]:+.3f}")
    print(f"ΔlnZ_BIC : NFW={delta_bic_NFW[-1]:+.3f},  BEC={delta_bic_BEC[-1]:+.3f},  BEC(rot)={delta_bic_BEC_rot[-1]:+.3f}")

# Convert to arrays (you’ll have ~175 entries if you run all SPARC)
# ======================================================
# Pretty histograms for ΔlnZ (harmonic & BIC) vs gNFW
# ======================================================
import numpy as np
import matplotlib.pyplot as plt

# (1) ensure arrays are 1D and finite
def _clean(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

delta_harm_NFW     = _clean(delta_harm_NFW)
delta_harm_BEC     = _clean(delta_harm_BEC)
delta_harm_BEC_rot = _clean(delta_harm_BEC_rot)

delta_bic_NFW      = _clean(delta_bic_NFW)
delta_bic_BEC      = _clean(delta_bic_BEC)
delta_bic_BEC_rot  = _clean(delta_bic_BEC_rot)

# (2) common bins helper (same as yours, tiny pad added)
def _common_bins(*arrays, num=25, pct_lo=1.0, pct_hi=99.0):
    all_vals = np.concatenate([a for a in arrays if len(a) > 0])
    lo = np.percentile(all_vals, pct_lo)
    hi = np.percentile(all_vals, pct_hi)
    pad = 0.02*(hi - lo) if hi > lo else 1e-3
    return np.linspace(lo - pad, hi + pad, num=num)

def _annotate_right_of_zero(ax, arr, y=0.94, label=None, color="k"):
    if len(arr) == 0: return
    frac_pos = (arr > 0).mean()
    txt = f"{label}: {int((arr > 0).sum())}/{len(arr)} > 0  ({100*frac_pos:0.1f}%)"
    ax.text(0.99, y, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=color)

def _draw_hist(ax, data, bins, color, label):
    ax.hist(data, bins=bins, alpha=0.55, label=label,
            edgecolor="white", linewidth=0.8, density=False, color=color)

def _median_line(ax, data, color):
    if len(data) == 0: return
    med = np.median(data)
    ax.axvline(med, color=color, lw=1.5, ls="-")

# color palette (matplotlib tab colors)
c_nfw = "tab:blue"
c_bec = "tab:orange"
c_rot = "tab:green"

# -------------------------------
# Harmonic lnZ
# -------------------------------
# --- Harmonic-mean lnZ deltas ---
bins_harm = _common_bins(delta_harm_NFW, delta_harm_BEC, delta_harm_BEC_rot, num=30)

fig, ax = plt.subplots(figsize=(8.4, 4.8))
_draw_hist(ax, delta_harm_NFW,     bins_harm, c_nfw, "NFW − gNFW")
_draw_hist(ax, delta_harm_BEC,     bins_harm, c_bec, "BEC (non-rot) − gNFW")
_draw_hist(ax, delta_harm_BEC_rot, bins_harm, c_rot, "BEC (rot) − gNFW")

ax.axvline(0.0, ls="--", color="k", lw=1.3)

_median_line(ax, delta_harm_NFW,     c_nfw)
_median_line(ax, delta_harm_BEC,     c_bec)
_median_line(ax, delta_harm_BEC_rot, c_rot)

ax.set_xlabel(r"$\Delta \ln Z_{\rm harm}$  (model $-$ gNFW)", fontsize=12)
ax.set_ylabel("Number of galaxies", fontsize=12)
ax.set_title(r"Across galaxies: $\Delta \ln Z_{\rm harm}$ relative to gNFW", fontsize=13)
ax.legend(frameon=False)
ax.grid(alpha=0.25, linestyle=":", linewidth=0.8)
fig.tight_layout()
fig.savefig("hist_delta_lnZ_harm_vs_gNFW_pretty.png", dpi=250)
plt.show()

# --- BIC-based lnZ deltas ---
bins_bic = _common_bins(delta_bic_NFW, delta_bic_BEC, delta_bic_BEC_rot, num=30)

fig, ax = plt.subplots(figsize=(8.4, 4.8))
_draw_hist(ax, delta_bic_NFW,      bins_bic, c_nfw, "NFW − gNFW")
_draw_hist(ax, delta_bic_BEC,      bins_bic, c_bec, "BEC (non-rot) − gNFW")
_draw_hist(ax, delta_bic_BEC_rot,  bins_bic, c_rot, "BEC (rot) − gNFW")

ax.axvline(0.0, ls="--", color="k", lw=1.3)

_median_line(ax, delta_bic_NFW,     c_nfw)
_median_line(ax, delta_bic_BEC,     c_bec)
_median_line(ax, delta_bic_BEC_rot, c_rot)

ax.set_xlabel(r"$\Delta \ln Z_{\rm BIC}$  (model $-$ gNFW)", fontsize=12)
ax.set_ylabel("Number of galaxies", fontsize=12)
ax.set_title(r"Across galaxies: $\Delta \ln Z_{\rm BIC}$ relative to gNFW", fontsize=13)
ax.legend(frameon=False)
ax.grid(alpha=0.25, linestyle=":", linewidth=0.8)
fig.tight_layout()
fig.savefig("hist_delta_lnZ_BIC_vs_gNFW_pretty.png", dpi=250)
plt.show()
