"""Recompute cached claim checks using only this publication package.

Outputs are diagnostics in generated/claims; the published dataset is read-only.
This is a cached numerical audit, not a fresh finite-p solve or a formal proof.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
out = root / "code/figure-reproduction/generated/claims"
out.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(root / "code"), str(root / "code/src")]
from scripts import generate_paper_figures as g
from large_p import evap_syk as es


def stats(a):
    a = np.asarray(a)
    f = a[np.isfinite(a)]
    return dict(shape=list(a.shape), finite=int(f.size), masked=int(a.size-f.size),
                min=float(f.min()), max=float(f.max()))


result = {}
p = g.manuscript_parameters()["finite_p"]
run, manifest = g.find_two_replica_run(p)
checks = []
for line in (run / "SHA256SUMS").read_text().splitlines():
    digest, rel = line.split("  ", 1)
    checks.append(hashlib.sha256((run / rel).read_bytes()).hexdigest() == digest)
result["checksums"] = dict(checked=len(checks), passed=sum(checks))
assert len(checks) == 51 and all(checks), "Published-data checksum mismatch"

import pandas as pd
rows = pd.read_csv(manifest)
errors = dict(beta_grid=0., antisymmetry=0.)
for row in rows.itertuples():
    obj = torch.load(manifest.parent / row.file, map_location="cpu", weights_only=False)
    c = obj["contour"]
    errors["beta_grid"] = max(errors["beta_grid"], abs(4*c["N_beta"]*c["dt_euclidean"]-c["beta"]))
    for k in ["G_chi", "G_disc", "G_twisted"]:
        a = obj[k].numpy()
        assert np.isfinite(a).all(), (row.file, k)
        errors["antisymmetry"] = max(errors["antisymmetry"], float(np.max(np.abs(a+a.T))))
result["saved_saddles"] = dict(count=len(rows), t_min=float(rows.t_f.min()),
    t_max=float(rows.t_f.max()), max_errors=errors,
    converged={k:int(rows[k+"_converged"].sum()) for k in ["chi","disc","twisted"]},
    max_residuals={k:float(rows[k+"_residual_norm"].max()) for k in ["chi","disc","twisted"]})

action = g.compute_renyi2_action_curve(p)["action_table"]
action.to_csv(out / "cached-entropy.csv", index=False)
y = action.S2_over_Neta.to_numpy()
t = action.t_f.to_numpy()
minima = [dict(t=float(t[i]), value=float(y[i])) for i in range(1,len(y)-1)
          if t[i]>p.tev and y[i]<y[i-1] and y[i]<y[i+1]]
maxima = [dict(t=float(t[i]), value=float(y[i])) for i in range(1,len(y)-1)
          if t[i]>p.tev and y[i]>y[i-1] and y[i]>y[i+1]]
result["entropy"] = dict(stats(y), pre_evaporation_max_abs=float(np.max(np.abs(y[t<=p.tev]))),
    local_minima=minima, local_maxima=maxima, log2=float(np.log(2)),
    value_at_25=float(y[-1]), columns=list(action.columns))

mi = g.compute_renyi2_mi_heatmaps(p)
np.savez_compressed(out / "cached-mi.npz", Iq=mi["Iq"], Icl=mi["Icl"],
                    Knorm=mi["Knorm"], times=mi["tgrid"])
result["mutual_information"] = dict(Iq=stats(mi["Iq"]), Icl=stats(mi["Icl"]),
    theoretical_maximum=float(2*np.log(2)),
    max_fraction=float(np.nanmax(mi["Iq"])/(2*np.log(2))),
    note="One final time (25), first right-moving component, kernel norm cutoff 50; does not establish a time sweep.")

p = g.manuscript_parameters()["operator_size"]
recon = g.build_large_p_reconstruction(p)
sizes = {}
for sector in ["eta","chi"]:
    fn = getattr(es, "compute_size_operator_DeltaN"+sector+"_grid")
    delta = np.real(fn(recon["tgrid"],25.,p.large_p_dict(),invalid_value=0j))
    sizes[sector] = g.bulk_operator_size_matrix(recon["KR_all"],recon["is_legit"],delta,2)
v_index = list(recon["tgrid"]).index(8.)
pd.DataFrame(dict(u=recon["tgrid"],eta=sizes["eta"][:,v_index],
                  chi=sizes["chi"][:,v_index])).to_csv(out / "size-v8.csv",index=False)
result["operator_size"] = {k:dict(stats(v), at_u0_v8=float(v[0,v_index])) for k,v in sizes.items()}

result["analytic"] = {}
for group in ["operator_size","geometry"]:
    p = g.manuscript_parameters()[group]
    pars = p.large_p_dict()
    times = p.tgrid()
    C, RR, RL, expR, expL = es.evapSYK2pt(times,pars)
    past = np.flatnonzero(times<p.tev)
    future = np.flatnonzero(times>p.tev)
    pi = (past[:,None]*2+np.arange(2)).ravel()
    fi = (future[:,None]*2+np.arange(2)).ravel()
    cross = C.real[np.ix_(fi,pi)]
    sv = np.linalg.svd(cross,compute_uv=False)
    ref = (np.sin(p.eps())/np.cosh(p.a*p.tev*np.sin(p.eps())))**2
    a = p.a*np.sin(p.eps())
    x,z = 1.,2.
    rr_paper = -(np.sin(p.eps())/np.sinh(.5*a*(x-z)-1j*p.eps()))**2
    rl_paper = (np.sin(p.eps())/np.cosh(.5*a*(x+z)))**2
    analytic = dict(theta=p.eps(),
        thermal_angle_residual=float(20*.5*p.a*np.sin(p.eps())-(np.pi-2*p.eps())),
        pre_R_error=float(abs(es.RI(x,z,a,p.eps())-rr_paper)),
        pre_L_error=float(abs(es.LI(x,z,a,p.eps())-rl_paper)),
        gram_symmetry_error=float(np.max(np.abs(C.real-C.real.T))),
        gram_diagonal_error=float(np.max(np.abs(np.diag(C.real)-1))),
        cross_rank_relative_1e_10=int(np.sum(sv>sv[0]*1e-10)),
        cross_singular_values=sv[:8].tolist(),
        post_L_matching_magnitude=ref,
        post_L_bound_ratio_max=float(np.max(np.abs(expL[np.ix_(future,future)]))/ref))
    result["analytic"][group] = analytic

# Independently transcribe the disk formula in S6; this is a numerical check
# of the stated inequality, not a proof or a new model result.
rng=np.random.default_rng(20260923)
z1=np.sqrt(rng.uniform(0,.999,10000))*np.exp(2j*np.pi*rng.random(10000))
z2=np.sqrt(rng.uniform(0,.999,10000))*np.exp(2j*np.pi*rng.random(10000))
rho=rng.uniform(0,.999,10000)
h=(1-abs(z1)**2)*(1-abs(z2)**2)*abs(1+rho*z1)**2*abs(1+rho*z2)**2/abs(1+rho*z1+rho*z2.conj()+z1*z2.conj())**2
k=(rho+z1)/(1+rho*z1)
hnorm=(1-abs(z1)**2)*(1+abs(rho-k.conj())**2/(1-abs(k)**2))
result["S6_disk_identity"] = dict(samples=len(h), min=float(h.min()), max=float(h.max()),
    hardy_norm_identity_error=float(np.max(np.abs(hnorm-(1-rho**2*abs(z1)**2)))),
    upper_bound_max_excess=float(np.max(h-np.minimum(1-rho**2*abs(z1)**2,1-rho**2*abs(z2)**2))))

(out / "numeric-audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
