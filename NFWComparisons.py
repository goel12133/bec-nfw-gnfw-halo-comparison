# ======================================================
# NFW vs gNFW comparison across galaxies
# ======================================================

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
    #Choose the galaxies you want to run the model on ___________________________________________________________________________
    
}

num_warmup,num_samples,num_chains=1000,3000,2

for gal_name,filepath in galaxies.items():
    print(f"\n===== {gal_name} =====")
    r,vobs,e_vobs,vg,vd,vb,_,_=np.genfromtxt(filepath,unpack=True)
    data_params={'r':r,'vg':vg,'vd':vd,'vb':vb}; n_data=len(r)

    # Run gNFW
    sampler_g=numpyro.infer.MCMC(numpyro.infer.NUTS(model_gnfw,target_accept_prob=0.9,dense_mass=True),
                                 num_warmup=num_warmup,num_samples=num_samples,num_chains=num_chains,progress_bar=False)
    sampler_g.run(jax.random.PRNGKey(0),r,e_vobs,vobs,data_params)
    posterior_g=sampler_g.get_samples()

    # Run NFW
    sampler_n=numpyro.infer.MCMC(numpyro.infer.NUTS(model_nfw,target_accept_prob=0.9,dense_mass=True),
                                 num_warmup=num_warmup,num_samples=num_samples,num_chains=num_chains,progress_bar=False)
    sampler_n.run(jax.random.PRNGKey(1),r,e_vobs,vobs,data_params)
    posterior_n=sampler_n.get_samples()

    # Corner plot (now includes alpha row/col)
        # ----- Corner plot: include alpha axis, but hide NFW on alpha by using NaNs -----
    param_order = ["log_c", "log_mh", "log_mld", "log_mlb", "alpha"]
    labels = [r"$\log c$", r"$\log M_{\rm vir}$", r"$\log (M/L)_d$", r"$\log (M/L)_b$", r"$\alpha$"]

    # gNFW (real alpha)
    X_g = np.column_stack([np.asarray(posterior_g[p]) for p in param_order])

# NFW: same 4 params + alpha column filled with constant (α=1)
    alpha_dummy = np.ones(len(posterior_n["log_c"]))  # not NaN, use 1.0
    X_n = np.column_stack([
        np.asarray(posterior_n["log_c"]),
        np.asarray(posterior_n["log_mh"]),
        np.asarray(posterior_n["log_mld"]),
        np.asarray(posterior_n["log_mlb"]),
        alpha_dummy,
    ])


    def _thin(X, n_max=4000):
        if X.shape[0] <= n_max: return X
        idx = np.random.choice(X.shape[0], size=n_max, replace=False)
        return X[idx]

    X_g, X_n = _thin(X_g), _thin(X_n)

    # shared ranges so contours align (include alpha too)
    ranges = []
    for i in range(X_g.shape[1]):
        both = np.concatenate([X_g[:, i], X_n[:, i]])
        both = both[np.isfinite(both)]  # drop NaNs from NFW alpha
        lo, hi = np.percentile(both, [0.5, 99.5])
        pad = 0.02 * (hi - lo) if hi > lo else 1e-3
        ranges.append((lo - pad, hi + pad))

    fig = corner.corner(
        X_g,
        labels=labels,
        color="tab:red",
        bins=30,
        smooth=1.0,
        range=ranges,
        plot_datapoints=False,
        fill_contours=True,
        levels=[0.393, 0.865],
        title_kwargs={"fontsize": 10},
    )

    corner.corner(
        X_n,
        fig=fig,
        color="tab:blue",
        bins=30,
        smooth=1.0,
        range=ranges,
        plot_datapoints=False,
        fill_contours=False,
        levels=[0.393, 0.865],
        contour_kwargs={"linestyles": "--", "linewidths": 1.3},
    )

    handles = [
        mlines.Line2D([], [], color="tab:red",  lw=6, label="gNFW ($\\alpha$ free)"),
        mlines.Line2D([], [], color="tab:blue", lw=2, ls="--", label="NFW ($\\alpha=1$)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10)

    plt.suptitle(f"{gal_name}: gNFW vs NFW (with $\\alpha$ axis)", fontsize=12)
    plt.savefig(f"corner_gnfw_nfw_with_alpha_{gal_name}.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ----- Evidence block (keep this indented so it runs for EACH galaxy) -----
    keys_n = ["log_c","log_mh","log_mld","log_mlb"]
    s_n = _prep_samples(posterior_n, keys_n)
    loglikes_n = np.asarray(_batch_loglike(s_n, False, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params))
    bic_n, llhat_n = _bic_from_loglikes(loglikes_n, k=4, n=n_data)
    lnZ_harm_n     = _harmonic_lnZ_from_loglikes(loglikes_n)
    lnZ_bic_n      = -0.5 * bic_n

    keys_g = ["log_c","log_mh","log_mld","log_mlb","alpha"]
    s_g = _prep_samples(posterior_g, keys_g)
    loglikes_g = np.asarray(_batch_loglike(s_g, True, jnp.array(r), jnp.array(vobs), jnp.array(e_vobs), data_params))
    bic_g, llhat_g = _bic_from_loglikes(loglikes_g, k=5, n=n_data)
    lnZ_harm_g     = _harmonic_lnZ_from_loglikes(loglikes_g)
    lnZ_bic_g      = -0.5 * bic_g

    print(f"NFW : max logL={llhat_n:.3f}, BIC={bic_n:.3f}, lnZ_harm={lnZ_harm_n:.3f}, lnZ_BIC={lnZ_bic_n:.3f}")
    print(f"gNFW: max logL={llhat_g:.3f}, BIC={bic_g:.3f}, lnZ_harm={lnZ_harm_g:.3f}, lnZ_BIC={lnZ_bic_g:.3f}")
    print(f"ΔBIC={bic_g-bic_n:.3f}, ΔlnZ_harm={lnZ_harm_g-lnZ_harm_n:.3f}, ΔlnZ_BIC={lnZ_bic_g-lnZ_bic_n:.3f}")
