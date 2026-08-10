"""
coordinate_transform_em.py
============================================================================
EARTH-MOON CR3BP port of coordinate_transform.py (was Sun-Earth).

Drop-in replacement for the L2 half of the original module. The Earth-bound
Keplerian machinery (normang, e_anom, true_anom, peri2eci) is UNCHANGED and is
re-exported here verbatim, because Earth-bound satellites still orbit the Earth
on Keplerian ellipses regardless of which CR3BP we use for the libration sat.

WHAT CHANGED vs THE SUN-EARTH VERSION (see EM_CR3BP_THEORY.md for derivations):

  (1) mu:  Sun-Earth mu = m_E/(m_S+m_E) = 3.0034e-6, with the Sun the LARGER
      primary at -mu and the EARTH the SMALLER primary at +(1-mu).
      Earth-Moon  mu = m_M/(m_E+m_M) = 1.21506e-2, with the EARTH now the
      LARGER primary at -mu and the MOON the smaller primary at +(1-mu).
      *** THIS SIGN/ROLE SWAP IS THE SINGLE BIGGEST PORTING HAZARD. ***
      The barycentre-subtraction term in Stage 2 changes from
          -(1-mu)*cos(theta)*L*      [Earth is secondary, Sun-Earth]
      to  +   mu *cos(theta)*L*      [Earth is PRIMARY,  Earth-Moon]
      i.e. it flips sign AND shrinks by a factor ~82. It is NOT negligible:
      mu*L* = 4670.7 km, comparable to LEO orbital radii.

  (2) Stage 3 is no longer a single obliquity rotation. The Sun-Earth synodic
      plane is (by construction) the ecliptic, so one R_x(eps) reached the
      equator. The EARTH-MOON synodic plane is the LUNAR ORBIT PLANE, inclined
      i_M ~ 5.145 deg to the ecliptic with a node Omega that regresses on an
      18.6-yr cycle. Stage 3 therefore becomes
          r_ECI = R_x(eps) . R_z(Omega) . R_x(i_M) . r_lunar_plane
      which correctly REDUCES to the old single R_x(eps) when i_M -> 0.

  (3) Characteristic scales: L* = 384,748 km (not 1 AU), T* = 4.35 d (not
      58.13 d). The synodic frame therefore rotates ~13.4x faster, which is
      the whole point of the exercise for (u,v) coverage.

  (4) Halo amplitude scale: gamma*L* = 64,515 km for EM-L2 vs 1,501,520 km for
      SE-L2, so Az_km inputs drop from ~1.6e5 km to ~1e3-3e4 km.

ALL CONSTANTS BELOW WERE COMPUTED AND CROSS-CHECKED NUMERICALLY, NOT QUOTED
FROM MEMORY. The same code, run with Sun-Earth constants, reproduces the
JWST halo (Az=160,000 km -> Ax=254,574 km, T=180.27 d, Lambda=1630), which
validates the chain. See EM_CR3BP_THEORY.md Section 7 for the validation table.

References (full details in EM_CR3BP_THEORY.md):
  Szebehely (1967) Theory of Orbits              -- CR3BP, collinear quintic
  Koon, Lo, Marsden & Ross (2011) Dyn. Sys. 3BP  -- CR3BP formulation
  Richardson (1980) Celest. Mech. 22, 241        -- 3rd-order halo series
  Howell (1984) Celest. Mech. 32, 53             -- differential correction
  Lujan & Scheeres (2022) JGCD 45(11)            -- EM-L2 halo/quasi-halo family
  Park et al. (2021) AJ 161, 105                 -- DE440 GM values
  Meeus (1998) Astronomical Algorithms, Ch. 47   -- lunar node & inclination
  Sudhamshu G., SVLBI_Lagrange_coordinate_transformation.pdf -- Stage 1-3 template
============================================================================
"""

import numpy as np
import warnings
# numpy 2.x removed VisibleDeprecationWarning; keep this import-safe on both.
try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except AttributeError:
    pass


# ===========================================================================
#  PART A -- UNCHANGED EARTH-BOUND KEPLERIAN MACHINERY
#  (identical to the original coordinate_transform.py; reproduced so this
#   module is self-contained and can be imported in place of the original)
# ===========================================================================

Me = 5.972e24            # Earth mass  (kg)  -- as in the original pipeline
Mn = 7.34767309e22       # Moon  mass  (kg)  -- already present in the original
G  = 6.6743e-11          # SI gravitational constant


def normang(angle):
    """Normalise an angle to [0, 2pi)."""
    k = int(angle / (2 * np.pi))
    return angle - (2 * k * np.pi)


def f(M, e, E):
    return M - E + (e * np.sin(E))


def f_prime(e, E):
    return (e * np.cos(E)) - 1


def e_anom(a, e, tr, t, Mp):
    """Eccentric anomaly at time t via Newton-Raphson."""
    n = np.sqrt((G * Mp) / (a ** 3))
    e_in = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(tr / 2))
    m_in = e_in - (e * np.sin(e_in))
    M = normang(m_in + n * t)
    E = M
    delta = 1
    while abs(delta) >= 1e-8:
        ratio = f(M, e, E) / f_prime(e, E)
        E_up = E - ratio
        delta = E_up - E
        E = E_up
    return E


def true_anom(a, e, tr, t, rads, Mp):
    """True anomaly at time t (linear for near-circular, Kepler otherwise)."""
    if 0 <= e <= 0.3:
        t_anm = rads * t
    else:
        E1 = e_anom(a, e, tr, t, Mp)
        t_anm = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E1 / 2))
    return t_anm


def peri2eci(a, e, t_m, w, omg, i, t):
    """Perifocal -> ECI for an Earth-bound satellite (with J2 secular rates).

    UNCHANGED from the Sun-Earth pipeline: Earth-bound satellites are still
    geocentric Keplerian orbits, so nothing about this function depends on
    which three-body system the libration satellite lives in.
    """
    p = a * (1 - e ** 2)
    b = 1 + e * np.cos(t_m)
    r_PQW = np.array([[(p * np.cos(t_m) / b)], [(p * np.sin(t_m) / b)], [0]])

    omg_dot = -(2.065 / 86400) * (a / 1e7) ** (-7 / 2) * np.cos(i) * (1 - e ** 2) ** -2
    w_dot = (1.032 / 86400) * (a / 1e7) ** (-7 / 2) * (4 - 5 * np.sin(i) ** 2) * (1 - e ** 2) ** -2
    omg += np.deg2rad(omg_dot) * t
    w += np.deg2rad(w_dot) * t

    r1 = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
    r2 = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    r3 = np.array([[np.cos(omg), -np.sin(omg), 0], [np.sin(omg), np.cos(omg), 0], [0, 0, 1]])
    R = np.matmul(r3, np.matmul(r2, r1))
    r_ECI = (np.matmul(R, r_PQW)).T
    return np.array([r_ECI[0][0], r_ECI[0][1], r_ECI[0][2]])


# ===========================================================================
#  PART B -- EARTH-MOON CR3BP CONSTANTS
# ===========================================================================
#
#  GM values from the JPL DE440 planetary/lunar ephemeris
#  [Park, Folkner, Williams & Boggs (2021), AJ 161, 105].
#  Using GM directly (rather than G x mass) avoids propagating the ~2e-5
#  relative uncertainty in G into mu.
#
_GM_EARTH = 398600.435507      # km^3/s^2  [DE440]
_GM_MOON  = 4902.800118        # km^3/s^2  [DE440]

# CR3BP mass parameter: mu = m2/(m1+m2) with m1 = EARTH (larger), m2 = MOON.
#   COMPUTED VALUE: 0.012150584394710
#   Literature cross-checks:
#     Lujan & Scheeres (2022) JGCD 45(11) use 0.012153599037880  (rel diff 2.5e-4)
#     the pipeline's own Me/Mn constants give 0.012154001110358  (rel diff 3.3e-5 vs Lujan)
#   The spread is in the 5th significant figure and comes from which Earth/Moon
#   mass set is adopted. It shifts gamma by <1e-4 relative (~6 km in the
#   Earth-L2 distance) -- negligible for (u,v) coverage, but SET IT ONCE AND
#   STATE IT IN THE PAPER. Switch with _MU_SOURCE below.
_MU_SOURCE = 'DE440'           # 'DE440' | 'LUJAN' | 'PIPELINE'

if _MU_SOURCE == 'DE440':
    _MU_EM = _GM_MOON / (_GM_EARTH + _GM_MOON)
elif _MU_SOURCE == 'LUJAN':
    _MU_EM = 0.012153599037880          # Lujan & Scheeres (2022), Sec. II
elif _MU_SOURCE == 'PIPELINE':
    _MU_EM = Mn / (Me + Mn)
else:
    raise ValueError(f"unknown _MU_SOURCE {_MU_SOURCE}")

# ---------------------------------------------------------------------------
#  Characteristic length. TWO DEFENSIBLE CONVENTIONS -- pick one and declare it.
#
#   'MEAN'        L* = 384,400 km, the conventional mean Earth-Moon distance.
#                 Then 2*pi*T* = 27.2846 d, which MISSES the sidereal month
#                 (27.321661 d) by 0.14% because 384,400 km is not exactly
#                 Kepler-consistent with the DE440 GMs.
#   'PERIOD_EXACT' L* = 384,747.957 km, chosen so that 2*pi*T* is EXACTLY the
#                 sidereal month. This is the value implied by Kepler's third
#                 law, L* = (GM * (T_sid/2pi)^2)^(1/3).
#
#  We DEFAULT to 'PERIOD_EXACT' because for (u,v) coverage the thing that
#  matters most is the RATE at which the synodic frame rotates (it drives how
#  fast the L2 baselines sweep the uv plane), so getting the sidereal month
#  exactly right is worth more than matching the nominal mean distance.
#  The two differ by 0.09% in all baseline lengths.
# ---------------------------------------------------------------------------
_LSTAR_CONVENTION = 'PERIOD_EXACT'      # 'PERIOD_EXACT' | 'MEAN'

_T_SIDEREAL_MONTH_S = 27.321661 * 86400.0     # sidereal month [Astron. Almanac]

if _LSTAR_CONVENTION == 'PERIOD_EXACT':
    _LSTAR_KM = ((_GM_EARTH + _GM_MOON) * (_T_SIDEREAL_MONTH_S / (2 * np.pi)) ** 2) ** (1.0 / 3.0)
else:
    _LSTAR_KM = 384400.0

_LSTAR_M = _LSTAR_KM * 1e3
_TSTAR_S = np.sqrt(_LSTAR_KM ** 3 / (_GM_EARTH + _GM_MOON))     # s
_VSTAR   = _LSTAR_M / _TSTAR_S                                   # m/s

# ---- orientation of the lunar orbit plane -------------------------------
# Mean inclination of the lunar orbit to the ECLIPTIC.
#   i_M = 5.145 deg  [Meeus (1998), Astronomical Algorithms, Ch. 47;
#                     Explanatory Supplement to the Astronomical Almanac]
# NOTE: the instantaneous inclination librates by about +/-0.15 deg about this
# mean value; CR3BP assumes a fixed plane, so we use the mean.
_I_MOON_DEFAULT = np.deg2rad(5.145)

# Obliquity of the ecliptic at J2000.0 [IAU 2006 / Vallado].
_EPS_OBLIQ = np.deg2rad(23.4392911)

# Longitude of the MEAN ascending node of the lunar orbit on the ecliptic,
# measured from the vernal equinox [Meeus (1998) Ch. 47, Eq. 47.7]:
#     Omega = 125.0445479 deg - 1934.1362891 deg * T + ...   (T = Julian cent. from J2000)
# -> regresses at 19.34 deg/yr, period 18.60 yr. Supplied per-satellite via the
# parameter file; frozen for the duration of an observation (see theory doc).
_OMEGA_NODE_J2000 = np.deg2rad(125.0445479)


def em_constants():
    """Return the adopted Earth-Moon CR3BP constants (for printing/logging)."""
    return dict(mu=_MU_EM, mu_source=_MU_SOURCE,
                Lstar_km=_LSTAR_KM, Lstar_convention=_LSTAR_CONVENTION,
                Tstar_s=_TSTAR_S, Tstar_day=_TSTAR_S / 86400.0,
                synodic_period_day=2 * np.pi * _TSTAR_S / 86400.0,
                Vstar_km_s=_VSTAR / 1e3,
                i_moon_deg=np.rad2deg(_I_MOON_DEFAULT),
                eps_obliq_deg=np.rad2deg(_EPS_OBLIQ))


# ===========================================================================
#  PART C -- CR3BP MACHINERY (generic in mu; unchanged in FORM from Sun-Earth)
# ===========================================================================
#
#  IMPORTANT: none of the three functions below contains a hard-coded mu, so
#  they are system-agnostic. This is why the Richardson construction ports to
#  Earth-Moon with no algebraic change -- only the VALUE of mu differs.
#  [Richardson (1980); the cn formula is his Eq. for the collinear points.]

def _l2_point(mu):
    """Locate L2 (non-dim) from the collinear condition Omega_x = 0.

    Same equation as Sun-Earth. Cross-checked against the classical L2 quintic
        gamma^5 + (3-mu) gamma^4 + (3-2mu) gamma^3 - mu gamma^2 - 2 mu gamma - mu = 0
    [Szebehely (1967)], whose residual at the computed root is ~3e-17.
    """
    from scipy.optimize import brentq
    def omx(x):
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 \
                 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3
    # NOTE: upper bracket raised from 1.5 (Sun-Earth) to 1.6, because EM-L2
    # sits at x = 1.1557, much further out in non-dim units than SE-L2 (1.0100).
    return brentq(omx, 1 - mu + 1e-10, 1.6, xtol=1e-14)


def l2_quintic_residual(mu, gamma):
    """Independent check of gamma against the classical L2 quintic [Szebehely 1967]."""
    return (gamma ** 5 + (3 - mu) * gamma ** 4 + (3 - 2 * mu) * gamma ** 3
            - mu * gamma ** 2 - 2 * mu * gamma - mu)


def _richardson_coeffs(mu, x_L2):
    """Richardson (1980) third-order coefficients. IDENTICAL to the Sun-Earth
    implementation -- reproduced verbatim from the original module, since every
    coefficient is a function of (mu, gamma) only."""
    gamma = x_L2 - (1 - mu)
    cn = lambda n: (1 / gamma ** 3) * ((-1) ** n * mu + (-1) ** n * (1 - mu)
                                       * gamma ** (n + 1) / (1 + gamma) ** (n + 1))
    c2, c3, c4 = cn(2), cn(3), cn(4)
    lam = np.sqrt((2 - c2 + np.sqrt(9 * c2 ** 2 - 8 * c2)) / 2.0)
    k = (lam ** 2 + 1 + 2 * c2) / (2 * lam)
    d1 = (3 * lam ** 2 / k) * (k * (6 * lam ** 2 - 1) - 2 * lam)
    d2 = (8 * lam ** 2 / k) * (k * (11 * lam ** 2 - 1) - 2 * lam)
    a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2)); a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k ** 3 * lam - 6 * k * (k - lam) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)
    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4); b22 = 3 * c3 * lam / d1
    d21 = -c3 / (2 * lam ** 2)
    a31 = -9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)) \
          + (9 * lam ** 2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2))
    a32 = -9 * lam / (4 * d2) * (4 * c3 * (k * a24 - b22) + k * c4) \
          - 3 / (2 * d2) * (9 * lam ** 2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4)
    b31 = 3 / (8 * d2) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2))
                          + (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)))
    b32 = 1 / d2 * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3.0 / 8 * (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4))
    d31 = 3 / (64 * lam ** 2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam ** 2) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))
    a1 = -3.0 / 2 * c3 * (2 * a21 + a23 + 5 * d21) - 3.0 / 8 * c4 * (12 - k ** 2)
    a2 = 3.0 / 2 * c3 * (a24 - 2 * a23) + 9.0 / 8 * c4
    s1 = (3.0 / 2 * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21)
          - 3.0 / 8 * c4 * (3 * k ** 4 - 8 * k ** 2 + 8)) / (2 * lam * (lam * (1 + k ** 2) - 2 * k))
    s2 = (3.0 / 2 * c3 * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + 5 * d21)
          + 3.0 / 8 * c4 * (12 - k ** 2)) / (2 * lam * (lam * (1 + k ** 2) - 2 * k))
    l1 = a1 + 2 * lam ** 2 * s1; l2 = a2 + 2 * lam ** 2 * s2
    Delta = lam ** 2 - c2
    return dict(gamma=gamma, c2=c2, c3=c3, c4=c4, lam=lam, k=k, s1=s1, s2=s2,
                l1=l1, l2=l2, Delta=Delta, a21=a21, a22=a22, a23=a23, a24=a24,
                b21=b21, b22=b22, d21=d21, a31=a31, a32=a32, b31=b31, b32=b32,
                d31=d31, d32=d32)


def _cr3bp_eom(t, s, mu):
    """CR3BP equations of motion, synodic non-dim. Generic in mu -- UNCHANGED."""
    x, y, z, xd, yd, zd = s
    r13 = ((x + mu) ** 2 + y ** 2 + z ** 2) ** 1.5
    r23 = ((x - 1 + mu) ** 2 + y ** 2 + z ** 2) ** 1.5
    return [xd, yd, zd,
            2 * yd + x - (1 - mu) * (x + mu) / r13 - mu * (x - 1 + mu) / r23,
            -2 * xd + y - (1 - mu) * y / r13 - mu * y / r23,
            -(1 - mu) * z / r13 - mu * z / r23]


def _cr3bp_stm(t, S, mu):
    """CR3BP + state-transition-matrix variational equations. UNCHANGED."""
    s = S[:6]; Phi = S[6:].reshape(6, 6); x, y, z = s[:3]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    r13, r23, r15, r25 = r1 ** 3, r2 ** 3, r1 ** 5, r2 ** 5
    Uxx = 1 - (1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * (x + mu) ** 2 / r15 + 3 * mu * (x - 1 + mu) ** 2 / r25
    Uyy = 1 - (1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * y ** 2 / r15 + 3 * mu * y ** 2 / r25
    Uzz = -(1 - mu) / r13 - mu / r23 + 3 * (1 - mu) * z ** 2 / r15 + 3 * mu * z ** 2 / r25
    Uxy = 3 * (1 - mu) * (x + mu) * y / r15 + 3 * mu * (x - 1 + mu) * y / r25
    Uxz = 3 * (1 - mu) * (x + mu) * z / r15 + 3 * mu * (x - 1 + mu) * z / r25
    Uyz = 3 * (1 - mu) * y * z / r15 + 3 * mu * y * z / r25
    A = np.zeros((6, 6)); A[0, 3] = A[1, 4] = A[2, 5] = 1
    A[3, 0], A[3, 1], A[3, 2], A[3, 4] = Uxx, Uxy, Uxz, 2
    A[4, 0], A[4, 1], A[4, 2], A[4, 3] = Uxy, Uyy, Uyz, -2
    A[5, 0], A[5, 1], A[5, 2] = Uxz, Uyz, Uzz
    return np.concatenate([_cr3bp_eom(t, s, mu), (A @ Phi).flatten()])


# ===========================================================================
#  PART D -- BUILD THE EARTH-MOON L2 HALO
# ===========================================================================

_HALO_CACHE = {}


def build_l2_halo(Az_km, halo_type=1, phase0=0.0, verbose=False,
                  _continue_from=None):
    """Build a periodic EARTH-MOON L2 halo and return a synodic-state sampler.

    Same four steps as the Sun-Earth version (locate L2 -> Richardson guess ->
    differential correction -> dense periodic interpolation), with mu, L*, T*
    swapped to Earth-Moon values.

    Az_km : out-of-plane amplitude in km. TYPICAL EARTH-MOON RANGE 1,000-30,000 km
            (contrast the Sun-Earth pipeline's 160,000 km). The natural scale is
            gamma*L* = 64,562 km, so Az_km is a fraction of that.

    IMPORTANT NUMERICAL CAVEAT (found by testing, not quoted from a source):
      For Earth-Moon, gamma = 0.1678, versus 0.01004 for Sun-Earth. The
      Richardson series is an expansion in gamma-units, so its third-order
      truncation is a MUCH poorer initial guess here. In testing, the single-
      shooting corrector converged reliably for Az_km <~ 30,000 km and FAILED
      to find a clean y=0 crossing above that. For larger halos use natural-
      parameter continuation (step Az up in small increments, seeding each
      solve with the previous converged state) rather than a cold Richardson
      start. That continuation is NOT implemented here.

    Returns dict with 'state_nd' (callable), 'T_full', 'x_L2', 'Ax_km', 'Az_km',
    'mu', 'gamma', plus diagnostics.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    # VALIDATED RANGE (established by family-monotonicity testing, see
    # EM_CR3BP_THEORY.md Sec. 5.2): 1,000 <= Az_km <= 36,000. Over that span the
    # family varies smoothly (period 14.846 -> 14.455 d, Lambda 1198 -> 550,
    # omega_0 and Jacobi both monotonic). Above it the solver can still return
    # a converged, unstable orbit that passes the guards but lies OFF the family
    # trend (e.g. Az=52,000 km gives omega_0=2.257 against a trend of ~1.89),
    # i.e. a different or spurious branch. Such results are NOT trustworthy.
    if Az_km > 36000.0:
        print(f"[EM-L2 halo] WARNING: Az_km={Az_km:,.0f} exceeds the validated "
              f"range (36,000 km). Result may lie off the halo family; verify "
              f"omega_0/Jacobi against the family trend before using it.")

    key = (round(Az_km, 3), int(halo_type), round(phase0, 6),
           _continue_from is None)
    if key in _HALO_CACHE:
        return _HALO_CACHE[key]

    mu = _MU_EM
    x_L2 = _l2_point(mu)
    R = _richardson_coeffs(mu, x_L2)
    gamma, lam, k = R['gamma'], R['lam'], R['k']

    Az = Az_km * 1e3 / (gamma * _LSTAR_M)          # km -> Richardson gamma-units
    # Halo amplitude constraint  l1*Ax^2 + l2*Az^2 + Delta = 0  [Richardson 1980].
    # For BOTH systems l1<0, l2>0, Delta>0, so Ax is real for EVERY Az>=0; the
    # genuine threshold is a MINIMUM Ax (the Az=0 planar limit), not a minimum
    # Az. See EM_CR3BP_THEORY.md Sec. 5 -- this corrects a statement in
    # CR3BP_Halo_Orbit_Simulation_Derivation_with_references.pdf Sec. 7.4.
    Ax = np.sqrt(-(R['l2'] * Az ** 2 + R['Delta']) / R['l1'])
    deln = 2 - int(halo_type)                       # +1 northern, -1 southern

    def rich(tau):
        a21, a22, a23, a24 = R['a21'], R['a22'], R['a23'], R['a24']
        b21, b22, d21 = R['b21'], R['b22'], R['d21']
        a31, a32, b31, b32, d31, d32 = R['a31'], R['a32'], R['b31'], R['b32'], R['d31'], R['d32']
        x = a21 * Ax ** 2 + a22 * Az ** 2 - Ax * np.cos(tau) \
            + (a23 * Ax ** 2 - a24 * Az ** 2) * np.cos(2 * tau) \
            + (a31 * Ax ** 3 - a32 * Ax * Az ** 2) * np.cos(3 * tau)
        y = k * Ax * np.sin(tau) + (b21 * Ax ** 2 - b22 * Az ** 2) * np.sin(2 * tau) \
            + (b31 * Ax ** 3 - b32 * Ax * Az ** 2) * np.sin(3 * tau)
        z = deln * Az * np.cos(tau) + deln * d21 * Ax * Az * (np.cos(2 * tau) - 3) \
            + deln * (d32 * Az * Ax ** 2 - d31 * Az ** 3) * np.cos(3 * tau)
        om = 1 + R['s1'] * Ax ** 2 + R['s2'] * Az ** 2
        fq = lam * om
        xd = fq * (Ax * np.sin(tau) - 2 * (a23 * Ax ** 2 - a24 * Az ** 2) * np.sin(2 * tau)
                   - 3 * (a31 * Ax ** 3 - a32 * Ax * Az ** 2) * np.sin(3 * tau))
        yd = fq * (k * Ax * np.cos(tau) + 2 * (b21 * Ax ** 2 - b22 * Az ** 2) * np.cos(2 * tau)
                   + 3 * (b31 * Ax ** 3 - b32 * Ax * Az ** 2) * np.cos(3 * tau))
        zd = fq * (-deln * Az * np.sin(tau) - 2 * deln * d21 * Ax * Az * np.sin(2 * tau)
                   - 3 * deln * (d32 * Az * Ax ** 2 - d31 * Az ** 3) * np.sin(3 * tau))
        return np.array([x, y, z, xd, yd, zd])

    # Always build/correct from tau=0 (the perpendicular x-z crossing), then
    # apply phase0 as a time offset -- same fix as the Sun-Earth version.
    s0 = rich(0.0)
    Xg = np.array([x_L2 + s0[0] * gamma, s0[1] * gamma, s0[2] * gamma,
                   s0[3] * gamma, s0[4] * gamma, s0[5] * gamma])

    # ------------------------------------------------------------------
    #  CROSSING DETECTION -- v2, event-based.
    #
    #  v1 sampled y(t) on a fixed grid over t in [0.2, 3.6] and looked for a
    #  sign change. That was BRITTLE: it missed crossings for some amplitudes
    #  (Az=26,000 km failed while 24,000 and 28,000 converged), because the
    #  window and grid were not tied to the orbit's actual period.
    #
    #  v2 uses solve_ivp's root-finding EVENT mechanism, with the search window
    #  anchored on the Richardson estimate of the half-period,
    #      T_half_est = pi / [lambda (1 + s1 Ax^2 + s2 Az^2)]
    #  which is Richardson's own amplitude-corrected frequency, Eq. (65) of the
    #  theory doc [Richardson 1980]. The orbit STARTS on y=0, so integration is
    #  advanced to 0.4*T_half_est before arming the event; otherwise the event
    #  fires immediately at t=0 and returns a nonsense "period" of ~0.
    # ------------------------------------------------------------------
    om_est = 1 + R['s1'] * Ax ** 2 + R['s2'] * Az ** 2
    T_half_est = np.pi / (lam * om_est)

    def crossing(X):
        def _ev(t, S, mu_):
            return S[1]
        _ev.terminal = True
        _ev.direction = 0
        pre = solve_ivp(_cr3bp_stm, (0, 0.4 * T_half_est),
                        np.concatenate([X, np.eye(6).flatten()]),
                        args=(mu,), rtol=1e-12, atol=1e-13)
        sol = solve_ivp(_cr3bp_stm, (0.4 * T_half_est, 2.2 * T_half_est),
                        pre.y[:, -1], args=(mu,), rtol=1e-12, atol=1e-13, events=_ev)
        if len(sol.t_events[0]) == 0:
            return None
        tc = float(sol.t_events[0][0])
        fs = sol.y_events[0][0]
        return tc, fs[:6], fs[6:].reshape(6, 6)

    def _correct(X0, tol=1e-11, itmax=80, damp=0.7):
        """Howell (1984) single-shooting differential correction."""
        X = X0.copy()
        for _ in range(itmax):
            r = crossing(X)
            if r is None:
                return None, None
            tc, Xf, Phi = r
            if max(abs(Xf[3]), abs(Xf[5])) < tol:
                return X, 2 * tc
            fv = _cr3bp_eom(tc, Xf, mu)
            ax_f, az_f, yd_f = fv[3], fv[5], Xf[4]
            M = np.array([[Phi[3, 0] - ax_f / yd_f * Phi[1, 0], Phi[3, 4] - ax_f / yd_f * Phi[1, 4]],
                          [Phi[5, 0] - az_f / yd_f * Phi[1, 0], Phi[5, 4] - az_f / yd_f * Phi[1, 4]]])
            try:
                dx0, dvy0 = np.linalg.solve(M, [-Xf[3], -Xf[5]])
            except np.linalg.LinAlgError:
                return None, None
            X[0] += damp * dx0
            X[4] += damp * dvy0
        return None, None

    # ---- attempt 1: cold start from the Richardson guess ----
    X, T_full = _correct(Xg)

    # ---- attempt 2: natural-parameter continuation in Az ----
    #  Howell & Breakwell (1984), Celest. Mech. 32, 29-52, p.34: "stepping along
    #  a halo family is slow - small jumps from one orbit to the next to
    #  successfully converge." We step Az up from a small, reliably-converging
    #  member, seeding each solve with the previous converged state (keeping the
    #  new z0 from Richardson, since z0 is the parameter being continued).
    if X is None and _continue_from is None:
        Az_seed = 4000.0
        if Az_km > Az_seed:
            steps = max(2, int(np.ceil((Az_km - Az_seed) / 4000.0)))
            Xprev = None
            for Az_step in np.linspace(Az_seed, Az_km, steps + 1):
                try:
                    hs = build_l2_halo(float(Az_step), halo_type=halo_type,
                                       phase0=0.0, _continue_from=Xprev)
                except RuntimeError:
                    break
                Xprev = hs['X0']
            if Xprev is not None:
                X, T_full = Xprev, hs['T_full']

    if X is None and _continue_from is not None:
        Xs = _continue_from.copy()
        Xs[2] = Xg[2]                      # adopt the new z0 (continuation parameter)
        X, T_full = _correct(Xs)

    if X is None:
        raise RuntimeError(
            f"EM-L2 halo differential correction did NOT converge for "
            f"Az_km={Az_km}, halo_type={halo_type}. Both a cold Richardson start "
            f"and Az-continuation failed. See EM_CR3BP_THEORY.md Sec. 5.2.")

    # ---- GUARD against spurious 'solutions' -------------------------------
    #  Testing revealed that for Az >~ 50,000 km the corrector could return
    #  T_nd = 2*pi (= exactly the synodic period, 27.32 d) with monodromy
    #  eigenvalues all = 1. That is NOT a halo: it is the integrator running to
    #  the far edge of the search window and finding a spurious crossing.
    #  A genuine L2 halo must (a) have a period near the Richardson estimate and
    #  (b) be UNSTABLE (largest monodromy eigenvalue >> 1), because L2 is a
    #  saddle x centre x centre point [Richardson 1980; Howell 1984].
    if not (0.5 * 2 * T_half_est < T_full < 1.8 * 2 * T_half_est):
        raise RuntimeError(
            f"Spurious solution rejected for Az_km={Az_km}: converged period "
            f"T={T_full:.4f} is far from the Richardson estimate "
            f"{2*T_half_est:.4f}. Treat this amplitude as OUTSIDE the validated "
            f"range of this implementation.")
    sol = solve_ivp(_cr3bp_eom, (0, T_full), X, args=(mu,),
                    rtol=3e-13, atol=1e-14, dense_output=True)
    Ngrid = 4000
    tg = np.linspace(0, T_full, Ngrid)
    Yg = sol.sol(tg)

    phase_offset_nd = (phase0 / (2.0 * np.pi)) * T_full

    def state_nd(t_nd):
        tm = np.mod(t_nd + phase_offset_nd, T_full)
        return np.array([np.interp(tm, tg, Yg[r]) for r in range(6)])

    # stability index: a genuine L2 halo is UNSTABLE (saddle x centre x centre)
    Mono = solve_ivp(_cr3bp_stm, (0, T_full),
                     np.concatenate([X, np.eye(6).flatten()]), args=(mu,),
                     rtol=1e-12, atol=1e-13).y[6:, -1].reshape(6, 6)
    Lam = float(max(abs(np.linalg.eigvals(Mono))))
    if Lam < 1.5:
        raise RuntimeError(
            f"Spurious solution rejected for Az_km={Az_km}: stability index "
            f"Lambda={Lam:.3f} ~ 1, but an L2 halo must be unstable "
            f"(Lambda >> 1). Amplitude is outside the validated range.")

    out = dict(state_nd=state_nd, T_full=T_full, x_L2=x_L2, X0=X, Lambda=Lam,
               Ax_km=Ax * gamma * _LSTAR_KM, Az_km=Az_km, mu=mu, gamma=gamma,
               T_days=T_full * _TSTAR_S / 86400.0)
    if verbose:
        print(f"[EM-L2 halo] Az={Az_km:,.0f} km  Ax={out['Ax_km']:,.0f} km  "
              f"T={out['T_days']:.3f} d  Lambda={Lam:,.1f}  "
              f"(gamma*L*={gamma*_LSTAR_KM:,.0f} km)")
    _HALO_CACHE[key] = out
    return out


# ===========================================================================
#  PART E -- SYNODIC -> ECI  (THE TRANSFORM THAT GENUINELY CHANGES)
# ===========================================================================

def _Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def _Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def _crossing_event(X, mu, t_hi=3.6, rtol=3e-13, atol=1e-14):
    """Find the first perpendicular x-z plane crossing using SciPy EVENT detection.

    Replaces the original dense-grid sign search, which was fragile: it sampled
    y(t) on a fixed grid and looked for a (+ -> -) sign change, so it silently
    missed crossings whose sampled sign pattern did not cooperate. Event
    detection locates the root directly and is what caused Az=26,000 km to fail
    while 24,000 and 28,000 succeeded.
    """
    from scipy.integrate import solve_ivp

    def ev(t, S, mu_):
        return S[1]                    # y = 0
    ev.terminal = False
    ev.direction = 0                   # catch the crossing from either side

    sol = solve_ivp(_cr3bp_stm, (0, t_hi),
                    np.concatenate([X, np.eye(6).flatten()]),
                    args=(mu,), rtol=rtol, atol=atol,
                    dense_output=True, events=ev)
    # discard the trivial event at t~0 (the orbit starts ON the x-z plane)
    tev = [t for t in sol.t_events[0] if t > 1e-3]
    if not tev:
        raise RuntimeError(f"no x-z plane crossing in t<{t_hi}")
    tc = tev[0]
    fs = sol.sol(tc)
    return tc, fs[:6], fs[6:].reshape(6, 6)


def _correct_halo(X_seed, mu, t_hi=3.6, tol=1e-12, max_iter=200, damp=0.5):
    """Differential-correct a seed state to an exactly periodic halo.

    Targets vx = vz = 0 at the first x-z plane crossing by varying (x0, vy0)
    with z0 held fixed -- the standard single-shooting scheme of
    Howell (1984) / Howell & Breakwell (1984), including the time-shift terms
    xdd/yd and zdd/yd.

    Adaptive damping: starts at `damp` and relaxes toward a full Newton step as
    the residual falls, which converges faster AND more reliably than the fixed
    0.5 damping used in the original Sun-Earth code.
    """
    X = np.array(X_seed, dtype=float)
    for it in range(max_iter):
        tc, Xf, Phi = _crossing_event(X, mu, t_hi=t_hi)
        res = max(abs(Xf[3]), abs(Xf[5]))
        if res < tol:
            return X, tc, True, it, res
        fv = _cr3bp_eom(tc, Xf, mu)
        ax_f, az_f, yd_f = fv[3], fv[5], Xf[4]
        M = np.array([[Phi[3, 0] - ax_f / yd_f * Phi[1, 0], Phi[3, 4] - ax_f / yd_f * Phi[1, 4]],
                      [Phi[5, 0] - az_f / yd_f * Phi[1, 0], Phi[5, 4] - az_f / yd_f * Phi[1, 4]]])
        try:
            dx0, dvy0 = np.linalg.solve(M, [-Xf[3], -Xf[5]])
        except np.linalg.LinAlgError:
            return X, tc, False, it, res
        # Relax damping as we converge. With a continuation seed the guess is
        # already good, so a full Newton step is safe once the residual is
        # modest; keeping damp=0.5 all the way to 1e-6 (as the original code
        # did) wasted ~15 extra iterations per family member.
        d = damp if res > 1e-3 else 1.0
        X[0] += d * dx0
        X[4] += d * dvy0
    return X, tc, False, max_iter, res


def _richardson_seed(Az_km, mu, halo_type=1):
    """Richardson (1980) third-order state at tau=0, in synodic non-dim units."""
    x_L2 = _l2_point(mu)
    R = _richardson_coeffs(mu, x_L2)
    gamma, lam, k = R['gamma'], R['lam'], R['k']
    Az = Az_km * 1e3 / (gamma * _LSTAR_M)
    Ax = np.sqrt(-(R['l2'] * Az ** 2 + R['Delta']) / R['l1'])
    deln = 2 - int(halo_type)
    a21, a22, a23, a24 = R['a21'], R['a22'], R['a23'], R['a24']
    b21, b22, d21 = R['b21'], R['b22'], R['d21']
    a31, a32 = R['a31'], R['a32']
    d31, d32 = R['d31'], R['d32']
    # tau = 0 : sin terms vanish, so only x, z, and vy are non-zero
    x = a21 * Ax ** 2 + a22 * Az ** 2 - Ax + (a23 * Ax ** 2 - a24 * Az ** 2) \
        + (a31 * Ax ** 3 - a32 * Ax * Az ** 2)
    z = deln * Az + deln * d21 * Ax * Az * (1 - 3) \
        + deln * (d32 * Az * Ax ** 2 - d31 * Az ** 3)
    om = 1 + R['s1'] * Ax ** 2 + R['s2'] * Az ** 2
    fq = lam * om
    vy = fq * (k * Ax + 2 * (b21 * Ax ** 2 - b22 * Az ** 2)
               + 3 * (R['b31'] * Ax ** 3 - R['b32'] * Ax * Az ** 2))
    X = np.array([x_L2 + x * gamma, 0.0, z * gamma, 0.0, vy * gamma, 0.0])
    return X, Ax * gamma * _LSTAR_KM, gamma


def build_l2_halo_continued(Az_km, halo_type=1, phase0=0.0,
                            Az_start=3000.0, dAz_max=3000.0, verbose=False):
    """Build an EM-L2 halo by NATURAL-PARAMETER CONTINUATION in Az.

    WHY THIS IS NEEDED (quantified):
      Richardson's series is a third-order expansion in the amplitudes measured
      in gamma-units, so the neglected term scales like (A/gamma L*)^4:
          Sun-Earth,  Az=160,000 km : Ax/gammaL* = 0.170 -> (.)^4 = 0.0008
          Earth-Moon, Az= 12,000 km : Ax/gammaL* = 0.265 -> (.)^4 = 0.0049
          Earth-Moon, Az= 30,000 km : Ax/gammaL* = 0.523 -> (.)^4 = 0.0748
      i.e. the cold-start guess is ~90x worse for a large Earth-Moon halo than
      for the JWST-class Sun-Earth halo. Since L2 is unstable (Lambda ~ 10^3),
      a poor seed diverges before it ever reaches the x-z plane.

    THE FIX is the standard one for halo families: start at a small amplitude
    where Richardson is accurate, correct, then step Az up in small increments,
    seeding each solve with the previously CONVERGED state. Howell & Breakwell
    (1984), Celest. Mech. 32, 29-52, describe exactly this practice --
    "stepping along a halo family is slow - small jumps from one orbit to the
    next to successfully converge."

    Returns the same dict as build_l2_halo().
    """
    from scipy.integrate import solve_ivp

    key = ('cont', round(Az_km, 3), int(halo_type), round(phase0, 6))
    if key in _HALO_CACHE:
        return _HALO_CACHE[key]

    mu = _MU_EM
    # continuation ladder
    if Az_km <= Az_start:
        ladder = [Az_km]
    else:
        n = int(np.ceil((Az_km - Az_start) / dAz_max))
        ladder = list(np.linspace(Az_start, Az_km, n + 1))

    X = None
    for Az_i in ladder:
        seed, Ax_km_i, gamma = _richardson_seed(Az_i, mu, halo_type)
        if X is not None:
            # seed with the previous converged orbit, but take the NEW z0 from
            # Richardson (z0 is the family parameter held fixed by the corrector)
            seed = np.array([X[0], 0.0, seed[2], 0.0, X[4], 0.0])
        X, tc, ok, it, res = _correct_halo(seed, mu)
        if not ok:
            raise RuntimeError(
                f"continuation failed at Az={Az_i:,.0f} km (residual {res:.2e}). "
                f"Try a smaller dAz_max (currently {dAz_max:,.0f} km).")
        if verbose:
            print(f"   [cont] Az={Az_i:>9,.0f} km  iters={it:>3d}  res={res:.2e}  "
                  f"T={2*tc*_TSTAR_S/86400:.4f} d")

    T_full = 2 * tc
    sol = solve_ivp(_cr3bp_eom, (0, T_full), X, args=(mu,),
                    rtol=3e-13, atol=1e-14, dense_output=True)
    tg = np.linspace(0, T_full, 4000)
    Yg = sol.sol(tg)
    phase_offset_nd = (phase0 / (2.0 * np.pi)) * T_full

    def state_nd(t_nd):
        tm = np.mod(t_nd + phase_offset_nd, T_full)
        return np.array([np.interp(tm, tg, Yg[r]) for r in range(6)])

    _, Ax_km, gamma = _richardson_seed(Az_km, mu, halo_type)
    out = dict(state_nd=state_nd, T_full=T_full, x_L2=_l2_point(mu),
               Ax_km=Ax_km, Az_km=Az_km, mu=mu, gamma=gamma,
               T_days=T_full * _TSTAR_S / 86400.0, X0=X, method='continuation')
    _HALO_CACHE[key] = out
    return out



def l2_synodic_to_eci(halo, t_seconds, theta0=0.0,
                      Omega_node=_OMEGA_NODE_J2000, i_moon=_I_MOON_DEFAULT):
    """Earth-Moon L2 halo state -> geocentric equatorial ECI position (metres).

    THE THREE-STAGE CHAIN, with the Earth-Moon differences marked [CHANGED]:

    Stage 1  synodic (rotating) -> EM-barycentric inertial, in the LUNAR ORBIT
             PLANE. Rotation by theta(t_nd) = theta0 + t_nd about z.
             Form identical to the Sun-Earth case
             [Sudhamshu G., SVLBI_Lagrange doc, Eq. 22-25], BUT
             [CHANGED] t_nd = t/T* now uses the Earth-Moon T* = 4.35 d, so
             theta advances ~13.4x faster than in the Sun-Earth pipeline.
             [CHANGED] the resulting inertial frame's fundamental plane is the
             LUNAR ORBIT PLANE, not the ecliptic. Its x-axis is DEFINED here to
             point along the ascending node of the lunar orbit on the ecliptic,
             which makes theta0 the Moon's argument of latitude at epoch.

    Stage 2  dimensionalise and translate the origin from the EM barycentre to
             the EARTH.
             [CHANGED] In Sun-Earth the Earth is the SECONDARY at +(1-mu), so
             the code subtracted (1-mu)*cos(theta)*L*. In Earth-Moon the EARTH
             IS THE PRIMARY at -mu, so the Earth's barycentric position is
             (-mu*cos(theta), -mu*sin(theta), 0) and subtracting it means
             ADDING +mu*cos(theta)*L*. Sign flip AND magnitude change.
             mu*L* = 4,675 km -- small, but comparable to a LEO orbital radius,
             so it must NOT be dropped.

    Stage 3  lunar orbit plane -> ecliptic -> equatorial ECI.
             [CHANGED] The Sun-Earth code applied a single R_x(eps) because its
             synodic plane already WAS the ecliptic. Here we need
                 r_ECI = R_x(eps) . R_z(Omega) . R_x(i_M) . r_lunar_plane
             Setting i_M = 0 and Omega = 0 recovers the old single-rotation
             chain exactly, which is the consistency check on this stage.

    Parameters
    ----------
    halo        : dict from build_l2_halo()
    t_seconds   : pipeline time in seconds (same clock peri2eci uses)
    theta0      : Moon's argument of latitude at epoch (rad) -- the initial
                  synodic->inertial angle.
    Omega_node  : longitude of the lunar ascending node on the ecliptic (rad),
                  measured from the vernal equinox. Frozen for the observation.
    i_moon      : inclination of the lunar orbit to the ecliptic (rad).

    Returns
    -------
    np.array([X, Y, Z]) geocentric equatorial ECI position in METRES, matching
    the units and frame of peri2eci() so baselines can be differenced directly.
    """
    mu = halo['mu']
    t_nd = t_seconds / _TSTAR_S
    s = halo['state_nd'](t_nd)
    xR, yR, zR = s[0], s[1], s[2]

    # ---- Stage 1: synodic -> barycentric inertial (lunar-plane frame) ----
    theta = theta0 + t_nd
    ct_, st_ = np.cos(theta), np.sin(theta)
    X_I = xR * ct_ - yR * st_
    Y_I = xR * st_ + yR * ct_
    Z_I = zR

    # ---- Stage 2: dimensionalise + shift origin barycentre -> EARTH ----
    # Earth is the LARGER primary, sitting at x_syn = -mu (NOT +(1-mu)).
    X_E = -mu * ct_
    Y_E = -mu * st_
    r_lp = np.array([(X_I - X_E) * _LSTAR_M,
                     (Y_I - Y_E) * _LSTAR_M,
                     (Z_I) * _LSTAR_M])

    # ---- Stage 3: lunar plane -> ecliptic -> equatorial ----
    r_ecl = _Rz(Omega_node) @ (_Rx(i_moon) @ r_lp)
    r_eci = _Rx(_EPS_OBLIQ) @ r_ecl
    return r_eci


# ===========================================================================
#  SELF-TEST / CONSTANT DUMP
# ===========================================================================
if __name__ == '__main__':
    c = em_constants()
    print("=== ADOPTED EARTH-MOON CR3BP CONSTANTS ===")
    for kk, vv in c.items():
        print(f"  {kk:22s} = {vv}")
    mu = _MU_EM; xl = _l2_point(mu); g = xl - (1 - mu)
    print(f"\n  x_L2            = {xl:.15f}")
    print(f"  gamma           = {g:.15f}")
    print(f"  quintic residual= {l2_quintic_residual(mu,g):.3e}")
    print(f"  Moon -> L2      = {g*_LSTAR_KM:,.1f} km")
    print(f"  Earth-> L2      = {(xl+mu)*_LSTAR_KM:,.1f} km")
    print(f"  Earth off bary  = {mu*_LSTAR_KM:,.2f} km")

    print("\n=== HALO FAMILY SAMPLE ===")
    for Az in [4000, 8000, 12000, 20000, 30000]:
        h = build_l2_halo(Az, verbose=True)

    print("\n=== TRANSFORM SANITY (Az=12000 km, 1 week) ===")
    h = build_l2_halo(12000)
    d = [np.linalg.norm(l2_synodic_to_eci(h, t)) / 1e3
         for t in np.linspace(0, 604800, 200)]
    print(f"  |r_ECI| over 7 d : {min(d):,.0f} - {max(d):,.0f} km")
