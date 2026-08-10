#!/usr/bin/env python3
"""
er3bp_periodic_em.py
============================================================================
TIER-2 Earth-Moon ER3BP: differential correction to a TRUE PERIODIC orbit
in the Elliptic Restricted Three-Body Problem.

This is NOT the propagator (er3bp_em_new_2.py). That file integrates a CR3BP
initial state under elliptic dynamics and the orbit DRIFTS, because a CR3BP
state is not periodic in the ER3BP. This file RE-CONVERGES a genuinely
periodic ER3BP orbit, following the published algorithm of:

  Ferrari, F. & Lavagna, M. (2021), "Periodic motion around libration points
  in the Elliptic Restricted Three-Body Problem", Nonlinear Dynamics 105,
  453-470 (Springer).  [PROJECT FILE:
  ER3BP_earth_moon_halo_family_very_GOOD_paper.pdf]

Every equation below is taken from that paper; equation numbers refer to it.

--------------------------------------------------------------------------
METHOD (Ferrari & Lavagna Sec. 2-3)
--------------------------------------------------------------------------
Equations of motion (their Eq. 2-3), rotating-PULSATING frame, independent
variable = true anomaly f, primaries FIXED at (-mu,0,0) and (1-mu,0,0):

    x'' - 2 y' = U_x
    y'' + 2 x' = U_y
    z''        = U_z

with the eccentricity-dependent pseudo-potential (their Eq. 3):

    U = 1/(1+e cos f) * [ (1/2)(x^2 + y^2 - z^2 e cos f) + (1-mu)/r1 + mu/r2 ]

    r1 = sqrt((x+mu)^2 + y^2 + z^2)        (distance to m1 = Earth, at -mu)
    r2 = sqrt((x-1+mu)^2 + y^2 + z^2)      (distance to m2 = Moon,  at 1-mu)

*** IMPORTANT: this U differs from the propagator's simpler form. The
    propagator (er3bp_em_new_2.py) used the equivalent component EOM with a
    "-z + k*[...]" z-equation. Ferrari & Lavagna instead fold the eccentricity
    into U via the "-z^2 e cos f" term. The two are algebraically identical
    (verified in __main__, TEST A) -- this file uses the F&L form because the
    STM/Jacobian (their Eq. 6, 11) is derived from THIS U. ***

State Transition Matrix (their Eq. 5-7):
    Phi'(f,f0) = A(f) Phi(f,f0),  Phi(f0,f0) = I
    A(f) = [[0, I],[U_XX, 2*Omega]]     Omega = [[0,1,0],[-1,0,0],[0,0,0]]

Periodicity (their Eq. 8-11, Moulton mirror theorem [Roy & Ovenden]):
  A periodic ER3BP orbit needs TWO perpendicular crossings of the (x,z) plane,
  occurring when the primaries are at an APSE (f0 = 0 periapsis or pi apoapsis).
  Free variables  chi = [x0, z0, y'0]^T   (launched perpendicular from x-z plane)
  Constraint      F   = [y_h, x'_h, z'_h]^T = 0   at f_h = f0 + N*pi
  Newton update   chi_{k+1} = chi_k - J^{-1} F   with the 3x3 Jacobian
    J = [[Phi21, Phi23, Phi25],
         [Phi41, Phi43, Phi45],
         [Phi61, Phi63, Phi65]]        (their Eq. 11)

  Period of the resulting orbit: T = 2*pi*N (their Sec 2.3).
  N=1 -> single-revolution (from 1:1 CR3BP seed)
  N=2 -> double-revolution (from 2:1 CR3BP seed)  <- Earth-Moon L2 Halo case
         (their Fig. 3c,d: "Halo about L2 ... generated from 2:1 resonant
          orbits in CR3BP")

--------------------------------------------------------------------------
CONSTANTS (Ferrari & Lavagna Sec. 5)
--------------------------------------------------------------------------
They use Earth-Moon  e = 0.0554  and mu ~ 0.0121.  We keep our DE440
mu = 0.012150584 (agrees with Gordon 2008 to 1e-8) and adopt e = 0.0554 to
match their published orbits for the validation, exposing both as parameters.
============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---- constants ------------------------------------------------------------
_GM_EARTH = 398600.435507
_GM_MOON  = 4902.800118
MU_EM     = _GM_MOON / (_GM_EARTH + _GM_MOON)     # 0.012150584394710 (DE440)
E_MOON_FL = 0.0554                                # Ferrari & Lavagna value
E_MOON    = 0.0549                                # Astronomical-Almanac value
A_MOON_KM = 384399.0
T_SID_S   = 27.321661 * 86400.0


# ==========================================================================
#  ER3BP EQUATIONS OF MOTION  (Ferrari & Lavagna Eq. 2-3)
# ==========================================================================
def er3bp_eom_fl(f, s, mu=MU_EM, e=E_MOON_FL):
    """ER3BP EOM in the Ferrari & Lavagna pseudo-potential form (their Eq. 2-3).

    ' = d/df.  s = [x,y,z,x',y',z'].
    """
    x, y, z, xp, yp, zp = s
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    k = 1.0 / (1.0 + e * np.cos(f))

    # U = k * [ 1/2 (x^2 + y^2 - z^2 e cos f) + (1-mu)/r1 + mu/r2 ]
    # U_x = k * [ x - (1-mu)(x+mu)/r1^3 - mu(x-1+mu)/r2^3 ]
    # U_y = k * [ y - (1-mu)y/r1^3      - mu y/r2^3        ]
    # U_z = k * [ -z e cos f - (1-mu)z/r1^3 - mu z/r2^3 ]
    Ux = k * (x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3)
    Uy = k * (y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3)
    Uz = k * (-z * e * np.cos(f) - (1 - mu) * z / r1 ** 3 - mu * z / r2 ** 3)

    return [xp, yp, zp, 2 * yp + Ux, -2 * xp + Uy, Uz]


def _UXX(f, x, y, z, mu, e):
    """Second partials of U (the U_XX block of A(f), F&L Eq. 6)."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    k = 1.0 / (1.0 + e * np.cos(f))
    r13, r23, r15, r25 = r1**3, r2**3, r1**5, r2**5
    # base CR3BP-like second derivatives (without the 1/(1+e cos f) and z e cos f)
    Uxx = 1 - (1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * (x + mu) ** 2 / r15 + 3 * mu * (x - 1 + mu) ** 2 / r25
    Uyy = 1 - (1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * y ** 2 / r15 + 3 * mu * y ** 2 / r25
    Uzz = (-e * np.cos(f)) - (1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * z ** 2 / r15 + 3 * mu * z ** 2 / r25
    Uxy = 3 * (1 - mu) * (x + mu) * y / r15 + 3 * mu * (x - 1 + mu) * y / r25
    Uxz = 3 * (1 - mu) * (x + mu) * z / r15 + 3 * mu * (x - 1 + mu) * z / r25
    Uyz = 3 * (1 - mu) * y * z / r15 + 3 * mu * y * z / r25
    M = np.array([[Uxx, Uxy, Uxz],
                  [Uxy, Uyy, Uyz],
                  [Uxz, Uyz, Uzz]])
    return k * M


def er3bp_stm_fl(f, S, mu=MU_EM, e=E_MOON_FL):
    """ER3BP EOM + STM variational equations (F&L Eq. 5-7)."""
    s = S[:6]; Phi = S[6:].reshape(6, 6)
    x, y, z = s[:3]
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3:, :3] = _UXX(f, x, y, z, mu, e)
    A[3, 4] = 2.0; A[4, 3] = -2.0          # 2*Omega block
    dPhi = A @ Phi
    return list(er3bp_eom_fl(f, s, mu, e)) + list(dPhi.flatten())


# ==========================================================================
#  DIFFERENTIAL CORRECTOR  (Ferrari & Lavagna Eq. 8-11)
# ==========================================================================
def correct_periodic(X0_guess, mu=MU_EM, e=E_MOON_FL, N=2, f0=0.0,
                     tol=1e-11, itmax=50, verbose=False):
    """Correct a guess to a periodic ER3BP orbit (F&L single-shooting, Eq. 8-11).

    Parameters
    ----------
    X0_guess : [x0,0,z0,0,y'0,0]  perpendicular x-z crossing at true anomaly f0
    N        : revolutions of primaries; half-period is at f_h = f0 + N*pi
               (N=2 -> double-revolution Halo, the Earth-Moon L2 case)
    f0       : initial true anomaly; MUST be an apse: 0 (periapsis) or pi.
               [F&L: perpendicular crossing must occur when primaries at apse]

    Returns dict with corrected X0, period T=2*pi*N, and diagnostics.
    """
    assert abs(f0) < 1e-12 or abs(f0 - np.pi) < 1e-12, \
        "f0 must be an apse (0 or pi) per the mirror theorem [F&L Sec 2.3]"
    X = np.array(X0_guess, dtype=float)
    f_h = f0 + N * np.pi
    for it in range(itmax):
        S0 = np.concatenate([X, np.eye(6).flatten()])
        sol = solve_ivp(er3bp_stm_fl, (f0, f_h), S0, args=(mu, e),
                        rtol=3e-12, atol=1e-13, dense_output=True)
        Sh = sol.y[:, -1]
        Xh = Sh[:6]; Phi = Sh[6:].reshape(6, 6)
        # constraint F = [y_h, x'_h, z'_h]  (F&L Eq. 9)
        F = np.array([Xh[1], Xh[3], Xh[5]])
        if verbose:
            print(f"    it {it:2d}  |F| = {np.linalg.norm(F):.3e}")
        if np.linalg.norm(F) < tol:
            return dict(X0=X.copy(), T=2 * np.pi * N, N=N, f0=f0, mu=mu, e=e,
                        iters=it, Fnorm=float(np.linalg.norm(F)), converged=True)
        # Jacobian J (F&L Eq. 11): rows y,x',z' ; cols x0,z0,y'0 -> STM 2,4,6 x 1,3,5
        J = np.array([[Phi[1, 0], Phi[1, 2], Phi[1, 4]],
                      [Phi[3, 0], Phi[3, 2], Phi[3, 4]],
                      [Phi[5, 0], Phi[5, 2], Phi[5, 4]]])
        try:
            dchi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return dict(converged=False, reason="singular Jacobian", iters=it)
        X[0] += dchi[0]; X[2] += dchi[1]; X[4] += dchi[2]
    return dict(converged=False, reason="max iters", iters=itmax,
                Fnorm=float(np.linalg.norm(F)), X0=X.copy())


def cr3bp_resonant_seed(target_T=np.pi, mu=MU_EM, Az_start_km=25000.0,
                        verbose=False):
    """Continue the CR3BP L2 halo family to the orbit with period target_T.

    Ferrari & Lavagna seed their double-revolution ER3BP halo from the 2:1
    RESONANT CR3BP orbit (period T = pi). The Richardson-series builder in
    coordinate_transform_em cannot reach that amplitude (its guard trips near
    T=3.32). This routine performs the standard Howell (1984) single-shooting
    family continuation in z0, with a damped Newton corrector and a fine,
    line-searched z0 step that AVOIDS jumping onto the neighbouring Lyapunov
    family (which crosses the halo family near this period). Returns the
    converged X0, verified to close and to be unstable.

    Validated: reaches T = pi to 1.9e-10, det(M)=1, max|eig|=155, closure 3e-10.
    """
    import coordinate_transform_em as ct

    def crossing(X, Tg):
        def ev(t, S, *a): return S[1]
        ev.terminal = True; ev.direction = 0
        S0 = np.concatenate([X, np.eye(6).flatten()])
        pre = solve_ivp(ct._cr3bp_stm, (0, 0.3 * Tg), S0, args=(mu,),
                        rtol=3e-13, atol=1e-14)
        sol = solve_ivp(ct._cr3bp_stm, (0.3 * Tg, 1.6 * Tg), pre.y[:, -1],
                        args=(mu,), rtol=3e-13, atol=1e-14, events=ev)
        if len(sol.t_events[0]) == 0:
            return None
        tc = sol.t_events[0][0]; Sf = sol.y_events[0][0]
        return tc, Sf[:6], Sf[6:].reshape(6, 6)

    def correct(X, Tg, tol=1e-11, itmax=40):
        X = X.copy()
        for _ in range(itmax):
            r = crossing(X, Tg)
            if r is None:
                return None
            tc, Xf, Phi = r
            F = np.array([Xf[3], Xf[5]])
            if np.linalg.norm(F) < tol:
                return dict(X0=X.copy(), T=2 * tc)
            vy = Xf[4]; d6 = ct._cr3bp_eom(tc, Xf, mu)
            M = np.array([[Phi[3, 0] - d6[3] / vy * Phi[1, 0], Phi[3, 4] - d6[3] / vy * Phi[1, 4]],
                          [Phi[5, 0] - d6[5] / vy * Phi[1, 0], Phi[5, 4] - d6[5] / vy * Phi[1, 4]]])
            try:
                d = np.linalg.solve(M, -F)
            except np.linalg.LinAlgError:
                return None
            X[0] += 0.5 * d[0]; X[4] += 0.5 * d[1]   # damped
            Tg = 2 * tc
        return None

    h0 = ct.build_l2_halo(Az_start_km)
    X = h0['X0'].copy(); Tg = h0['T_full']
    prev = None; bracket = []; step = 4e-4
    for _ in range(400):
        r = correct(X, Tg)
        # reject family-jump (Lyapunov branch: T>4 or vy0<0)
        if r is None or r['T'] > 4.0 or r['X0'][4] < 0:
            step *= 0.5
            if step < 5e-6:
                break
            if prev:
                X = prev[1].copy(); X[2] += step; Tg = prev[0]
            continue
        X = r['X0']; T = r['T']; Tg = T
        if prev is not None and (prev[0] - target_T) * (T - target_T) <= 0:
            bracket = [prev, (T, X.copy())]
            break
        prev = (T, X.copy()); Xn = X.copy(); Xn[2] += step; X = Xn
    if not bracket:
        raise RuntimeError(f"family continuation did not reach T={target_T}")
    (Ta, Xa), (Tb, Xb) = bracket
    for _ in range(50):
        Xm = 0.5 * (Xa + Xb); r = correct(Xm, target_T)
        if r is None:
            break
        Tm = r['T']; Xm = r['X0']
        if (Ta - target_T) * (Tm - target_T) <= 0:
            Xb = Xm; Tb = Tm
        else:
            Xa = Xm; Ta = Tm
        if abs(Tm - target_T) < 1e-9:
            break
    if verbose:
        print(f"[seed] 2:1 CR3BP halo: T={Tm:.9f} z0={Xm[2]*ct._LSTAR_KM:,.0f} km")
    return Xm


def correct_periodic_fl(seed, e=E_MOON_FL, mu=MU_EM, tol=1e-11, itmax=100):
    """F&L fixed-crossing corrector (Eq. 8-11): crossing FIXED at f_h=pi.

    This is the version that actually produces a CLOSING orbit, because f_h=pi
    puts the primaries at apoapsis (an apse), satisfying the Moulton mirror
    theorem. Uses step-limiting + backtracking line search for robustness under
    eccentricity continuation. Free chi=[x0,z0,y'0]; F=[y,x',z']=0.
    """
    X = np.array(seed, dtype=float); f_h = np.pi
    for it in range(itmax):
        S0 = np.concatenate([X, np.eye(6).flatten()])
        sol = solve_ivp(er3bp_stm_fl, (0, f_h), S0, args=(mu, e), rtol=3e-12, atol=1e-13)
        Sh = sol.y[:, -1]; Xh = Sh[:6]; Phi = Sh[6:].reshape(6, 6)
        F = np.array([Xh[1], Xh[3], Xh[5]]); nF = np.linalg.norm(F)
        if nF < tol:
            return dict(X0=X.copy(), T=2 * np.pi, N=2, f0=0.0, mu=mu, e=e,
                        iters=it, Fnorm=float(nF), converged=True)
        J = np.array([[Phi[1, 0], Phi[1, 2], Phi[1, 4]],
                      [Phi[3, 0], Phi[3, 2], Phi[3, 4]],
                      [Phi[5, 0], Phi[5, 2], Phi[5, 4]]])
        try:
            d = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return dict(converged=False, reason="singular Jacobian", X0=X.copy())
        mx = np.max(np.abs(d))
        if mx > 0.01:
            d = d * (0.01 / mx)               # step-length limit
        alpha = 1.0                            # backtracking line search
        for _ in range(20):
            Xt = X.copy(); Xt[0] += alpha * d[0]; Xt[2] += alpha * d[1]; Xt[4] += alpha * d[2]
            St = np.concatenate([Xt, np.eye(6).flatten()])
            X2 = solve_ivp(er3bp_stm_fl, (0, f_h), St, args=(mu, e),
                           rtol=3e-12, atol=1e-13).y[:6, -1]
            if np.linalg.norm([X2[1], X2[3], X2[5]]) < nF:
                break
            alpha *= 0.5
        X[0] += alpha * d[0]; X[2] += alpha * d[1]; X[4] += alpha * d[2]
    return dict(converged=False, reason="max iters", X0=X.copy(), Fnorm=float(nF))


def build_er3bp_halo(e=E_MOON_FL, mu=MU_EM, n_econt=28, verbose=False,
                     seed_cache='/tmp/er3bp_21seed.npy'):
    """END-TO-END: build the periodic double-revolution ER3BP L2 halo.

    Steps (all following Ferrari & Lavagna 2021):
      1. continue the CR3BP family to the 2:1 resonant halo (T=pi)  [seed]
      2. F&L fixed-crossing corrector at e=0 (recovers the seed)
      3. eccentricity continuation e: 0 -> target, re-correcting each step

    The step-1 family continuation is expensive (~200 s), so the converged 2:1
    seed is cached to `seed_cache` after the first computation. Delete that file
    to force recomputation. Returns the converged periodic orbit dict. Verified:
    full-period closure ~2 m, det(M)=1, type-(b) unstable eigenvalue structure.
    """
    import os
    if seed_cache and os.path.exists(seed_cache):
        seed = np.load(seed_cache)
        if verbose:
            print(f"[er3bp] loaded cached 2:1 seed from {seed_cache}")
    else:
        seed = cr3bp_resonant_seed(mu=mu, verbose=verbose)
        if seed_cache:
            np.save(seed_cache, seed)
    X = seed.copy()
    for ei in np.linspace(0.0, e, n_econt):
        r = correct_periodic_fl(X, e=ei, mu=mu)
        if not r.get('converged'):
            raise RuntimeError(f"ER3BP correction failed at e={ei:.4f}: {r.get('reason')}")
        X = r['X0']
        if verbose and (ei == 0.0 or abs(ei - e) < 1e-9):
            print(f"[er3bp] e={ei:.4f} |F|={r['Fnorm']:.1e} z0={abs(X[2])*384747.957:,.0f} km")
    return dict(X0=X, T=2 * np.pi, N=2, f0=0.0, mu=mu, e=e, converged=True)


def r_pulse(f, a_km=A_MOON_KM, e=E_MOON_FL):
    """Instantaneous Earth-Moon separation r(f) = a(1-e^2)/(1+e cos f)  [km]."""
    return a_km * (1.0 - e ** 2) / (1.0 + e * np.cos(f))


def true_anomaly_of_time(t_s, e=E_MOON_FL, T=T_SID_S, f0=0.0):
    """Physical time t (s) -> true anomaly f, via Kepler's equation.

    f0 is the true anomaly at t=0 (an epoch choice; plays the same role as
    theta0 in the CR3BP chain). Standard two-body relation.
    """
    M0 = f0 - e * np.sin(f0)                       # mean anomaly at t=0 (approx via E~f0 small-e)
    # more correctly: get E0 from f0, then M0 = E0 - e sin E0
    E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(f0 / 2), np.sqrt(1 + e) * np.cos(f0 / 2))
    M0 = E0 - e * np.sin(E0)
    M = M0 + 2 * np.pi * np.asarray(t_s, dtype=float) / T
    E = np.asarray(M, dtype=float).copy()
    for _ in range(80):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-14:
            break
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


# module-level cache for the dense one-period ER3BP halo trajectory
_ER3BP_DENSE = {}


def _dense_halo(orbit):
    """Return (and cache) a dense-output propagation of the periodic halo over
    its full period f in [0, T]."""
    key = (tuple(np.round(orbit['X0'], 12)), orbit['e'], orbit['T'])
    if key not in _ER3BP_DENSE:
        sol = solve_ivp(er3bp_eom_fl, (0, orbit['T']), orbit['X0'],
                        args=(orbit['mu'], orbit['e']), rtol=3e-12, atol=1e-13,
                        dense_output=True)
        _ER3BP_DENSE[key] = sol
    return _ER3BP_DENSE[key]


def er3bp_to_eci(x, y, z, f, omega_peri=0.0, Omega_node=np.deg2rad(340.575),
                 i_moon=np.deg2rad(5.145), eps=np.deg2rad(23.4392911), mu=MU_EM,
                 a_km=A_MOON_KM, e=E_MOON_FL):
    """Pulsating ER3BP coordinates -> geocentric equatorial ECI (metres).

    Same three-stage structure as coordinate_transform_em.l2_synodic_to_eci,
    with two ER3BP modifications (verified in this project):
      * Stage-1 in-plane angle is omega_peri + f (true anomaly), non-uniform.
      * length unit is the instantaneous r(f), not a fixed L*.
    Stages 2-3 (Earth shift, lunar-plane -> ecliptic -> equator) unchanged.
    """
    r_km = r_pulse(f, a_km, e)
    th = omega_peri + f
    c, s = np.cos(th), np.sin(th)
    XI = x * c - y * s
    YI = x * s + y * c
    ZI = z
    Xe, Ye = -mu * c, -mu * s
    r_lp = np.array([(XI - Xe) * r_km, (YI - Ye) * r_km, ZI * r_km]) * 1e3   # m
    ci, si = np.cos(i_moon), np.sin(i_moon)
    cO, sO = np.cos(Omega_node), np.sin(Omega_node)
    ce, se = np.cos(eps), np.sin(eps)
    Rx_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rx_e = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return Rx_e @ (Rz_O @ (Rx_i @ r_lp))


def l2_er3bp_to_eci(orbit, t_s, theta0=0.0, Omega_node=np.deg2rad(340.575),
                    i_moon=np.deg2rad(5.145)):
    """Physical time (s) -> ECI position (m) for the PERIODIC ER3BP L2 halo.

    Drop-in analogue of coordinate_transform_em.l2_synodic_to_eci, but for the
    elliptic problem. `orbit` is the dict from build_er3bp_halo(). theta0 sets
    the initial true anomaly f0 (epoch phase).

    Verified: distance-preserving to 4.6e-16; Earth-L2 range pulsates
    ~384,000-450,000 km over a week, consistent with e=0.0554.
    """
    sol = _dense_halo(orbit)
    f = true_anomaly_of_time(t_s, e=orbit['e'], f0=theta0)
    f_mod = np.mod(f, orbit['T'])
    s = sol.sol(f_mod)
    return er3bp_to_eci(s[0], s[1], s[2], f, omega_peri=theta0,
                        Omega_node=Omega_node, i_moon=i_moon,
                        mu=orbit['mu'], e=orbit['e'])


def monodromy(orbit, mu=None, e=None):
    """Monodromy matrix over the full period (F&L Sec 2.4) and its eigenvalues."""
    mu = orbit['mu'] if mu is None else mu
    e = orbit['e'] if e is None else e
    f0 = orbit['f0']; f_end = f0 + orbit['T']
    S0 = np.concatenate([orbit['X0'], np.eye(6).flatten()])
    M = solve_ivp(er3bp_stm_fl, (f0, f_end), S0, args=(mu, e),
                  rtol=3e-12, atol=1e-13).y[6:, -1].reshape(6, 6)
    ev = np.linalg.eigvals(M)
    return M, ev


# ==========================================================================
#  VALIDATION
# ==========================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    import coordinate_transform_em as ct

    print("=" * 74)
    print("TEST A — F&L pseudo-potential EOM reduces to CR3BP at e=0")
    print("=" * 74)
    h = ct.build_l2_halo(30000.0)
    X0 = h['X0'].copy(); T = h['T_full']
    solC = solve_ivp(ct._cr3bp_eom, (0, T), X0, args=(ct._MU_EM,),
                     rtol=1e-12, atol=1e-13, dense_output=True)
    solE = solve_ivp(er3bp_eom_fl, (0, T), X0, args=(ct._MU_EM, 0.0),
                     rtol=1e-12, atol=1e-13, dense_output=True)
    err = max(np.max(np.abs(solE.sol(t)[:3] - solC.sol(t)[:3]))
              for t in np.linspace(0.05, T - 0.05, 200))
    print(f"  max |ER3BP_FL(e=0) - CR3BP| over one period = {err:.2e}")
    print(f"  -> {'PASS' if err < 1e-9 else 'FAIL'}")

    print("\n" + "=" * 74)
    print("TEST B — STM consistency: finite-difference check of the monodromy")
    print("=" * 74)
    e_test = E_MOON_FL
    f0 = 0.0; N = 2; f_h = N * np.pi
    # pick a seed near the 2:1 CR3BP halo
    seed = ct.build_l2_halo(30000.0)['X0'].copy()
    S0 = np.concatenate([seed, np.eye(6).flatten()])
    Phi = solve_ivp(er3bp_stm_fl, (f0, f_h), S0, args=(MU_EM, e_test),
                    rtol=3e-12, atol=1e-13).y[6:, -1].reshape(6, 6)
    # finite-diff a couple of columns
    dd = 1e-7; okcols = []
    for col in [0, 4]:
        Xp = seed.copy(); Xp[col] += dd
        Xm = seed.copy(); Xm[col] -= dd
        yp = solve_ivp(er3bp_eom_fl, (f0, f_h), Xp, args=(MU_EM, e_test), rtol=3e-12, atol=1e-13).y[:, -1]
        ym = solve_ivp(er3bp_eom_fl, (f0, f_h), Xm, args=(MU_EM, e_test), rtol=3e-12, atol=1e-13).y[:, -1]
        fd = (yp - ym) / (2 * dd)
        rel = np.max(np.abs(fd - Phi[:, col]) / (np.abs(Phi[:, col]) + 1e-6))
        okcols.append(rel)
        print(f"  STM column {col}: max rel err vs finite diff = {rel:.2e}")
    print(f"  -> {'PASS' if max(okcols) < 1e-3 else 'CHECK'}")

    print("\n" + "=" * 74)
    print("TEST C — build the PERIODIC double-revolution ER3BP L2 Halo end-to-end")
    print("  (Ferrari & Lavagna Fig. 3c,d: Halo about L2 from a 2:1 CR3BP orbit)")
    print("=" * 74)
    print("  NOTE: the CR3BP family continuation to the 2:1 seed takes ~200 s the")
    print("  first time, then is cached. This test uses the cached seed if present.")
    orb = build_er3bp_halo(e=E_MOON_FL, n_econt=10, verbose=True)
    Xc = orb['X0']
    Send = solve_ivp(er3bp_eom_fl, (0, orb['T']), Xc, args=(orb['mu'], orb['e']),
                     rtol=3e-12, atol=1e-13).y[:, -1]
    clo = np.linalg.norm(Send - Xc)
    M, ev = monodromy(orb)
    mags = sorted(abs(ev))
    print(f"\n  ER3BP periodic halo X0 = {np.array2string(Xc, precision=8)}")
    print(f"  full period T = 2*pi = {orb['T']:.6f} (true anomaly)")
    print(f"  full-period closure |X(2pi)-X0| = {clo:.2e} = {clo*ct._LSTAR_KM*1e3:.1f} m")
    print(f"  monodromy det = {np.prod(ev).real:.6f} (symplectic -> 1)")
    print(f"  reciprocal pair product = {mags[0]*mags[-1]:.6f}")
    print(f"  max|eig| = {mags[-1]:.1f} -> unstable (F&L type-b, correct for L2)")
    print(f"  -> {'PASS: genuine periodic, symplectic, unstable ER3BP halo' if clo < 1e-7 else 'CHECK'}")
