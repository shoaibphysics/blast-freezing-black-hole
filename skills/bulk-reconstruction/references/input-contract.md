# Input and reconstruction contract

## Arrays

Supply a non-pickled NumPy `.npz` with `times` (real, finite, strictly increasing,
length `T >= 2`), `two_point` (full matrix), and optionally `lesser` for the
complex greater/lesser convention.

Use `(T*N,T*N)` in time-major order, or `(T,N,T,N)` with axes
`(time_i,flavor_a,time_j,flavor_b)`. These reshape directly into each other.
For `(T,T,N,N)`, transpose with `(0,2,1,3)` first. Do not infer layout from
coincident dimension sizes. Analytic inputs must be evaluated at both times and
all flavor pairs. Triangular, retarded, time-ordered or contour data need a
documented conversion; do not guess missing Wightman blocks or contact terms.

The grid need not be uniform or start at zero. Gate coordinates are
`t=(times[u]+times[v])/2`, `z=(times[v]-times[u])/2`. A nonuniform grid still defines
circuit ordering, but its variable step lengths must be retained in plots.

## Explicit `--kind` choices

Here i,j each include time and flavor; H is conjugate transpose, whereas `.T`
exchanges both time and flavor without conjugation.

| Kind | Supplied matrix | Conversion and checks |
| --- | --- | --- |
| `majorana-anticommutator` | `A_ij=<{chi_i,chi_j}>` | Real symmetric, PSD, equal-time blocks `I_N` |
| `majorana-wightman` | `W_ij=<chi_i chi_j>` | Hermitian PSD; `A=W+W.T=2 Re W` |
| `majorana-greater` | `G^>_ij=-i<chi_i chi_j>` | Anti-Hermitian, `iG^>` PSD; `A=i(G^>+(G^>).T)` |
| `majorana-normalized` | `CorM=2<chi_i chi_j>` | Hermitian PSD; `A=(CorM+CorM.T)/2` |
| `complex-anticommutator` | `A_ij=<{c_i,c_j^dagger}>` | Hermitian PSD, possibly complex, equal-time blocks `I_N` |
| `complex-greater-lesser` | `G^>_ij=-i<c_i c_j^dagger>` and `G^<_ij=+i<c_j^dagger c_i>` | Both anti-Hermitian, `iG^>` and `-iG^<` PSD; `A=i(G^>-G^<)` |

Hermitian-symmetrizing a Majorana Wightman matrix does NOT isolate the
anticommutator: its imaginary part still carries commutator information.
Conversely, `Re G^>` is the wrong part in the `-i` convention. Numerical checks
cannot distinguish a physically mislabeled matrix with the same symmetries.

The complex mode reconstructs the supplied anticommutator span. Superconducting
or Nambu problems needing particle-hole mixing must supply a closed operator
basis including anomalous blocks. Converting to Hermitian Majoranas generally
doubles the component count. Do not claim missing anomalous data are included.

## Validation and limitations

Default symmetry tolerance is `atol + rtol*max(abs(matrix))`, with `atol=1e-10`,
`rtol=1e-8`. Eigenvalue checks use a spectral-scale tolerance. The declared
normalization is `{chi_a,chi_b}=delta_ab` or `{c_a,c_b^dagger}=delta_ab`. Explicitly
convert alternatives such as `{gamma_a,gamma_b}=2 delta_ab` beforehand. A known
nonidentity equal-time metric requires a recorded, invertible per-time whitening
transformation on both axes. Singular equal-time metrics require selecting an
independent basis. Do not infer normalization from a failed check.

Only roundoff-scale symmetry/reality cleanup is allowed after passing checks.
Eigenvalues are not clipped and no ridge is added. Rank-deficient long intervals
can legitimately fail Cholesky. The backend has a Cholesky pivot floor `1e-8`
and a preliminary Frobenius gate-unitarity cutoff `1e-2`. The helper requires
unitarity and input/output whitening residuals below `--gate-tol` (default
`1e-6`), then enforces interval inclusion on surviving gates.

Dense kernels alone require `2*T^3*N^2*itemsize` bytes. The helper's default
limit is 2 GiB; change `--max-kernel-gib` explicitly for larger runs. Input arrays,
eigensolvers and workspaces use additional memory. Not saving kernels avoids
large output files but does not avoid their allocation inside the current engine.

## Outputs and gate convention

`reconstruction.npz`: `times`, scalar `N`, `A`, integer `gate_indices` `(G,2)`,
`gates` `(G,2N,2N)`, refined `is_legit` gate mask, and backend path/hash.
`--save-kernels` adds `KL_all`, `KR_all` shaped `(T,T,N,T*N)`. Never use values at
invalid intervals. The gate mask excludes `u=v`, which has boundary kernels but
no bulk gate.

The engine defines `O=stack(KL[u+1,v],KR[u,v])`,
`I=stack(KR[u,v-1],KL[u,v])`, `U=O A I^H`. The helper checks
`O A O^H=I_(2N)`, `I A I^H=I_(2N)`, and `U U^H=I_(2N)` before keeping a gate,
and reports rejection counts and residuals. It does not project gates onto the
nearest unitary.

`diagnostics.json` records convention, sizes, tolerances, validation residuals,
rank information, gate counts, and backend hash. Cholesky fixes an
ordering-dependent frame. Transform observables explicitly if their specified
frame differs; do not assume reconstructed components remain identical to
physical species after an undocumented basis change.
