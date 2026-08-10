"""
uvw_eval_em.py  --  EARTH-MOON variant of uvw_eval.py.
============================================================================
Identical to uvw_eval.py EXCEPT that the L2 satellite is propagated with the
Earth-Moon CR3BP module (coordinate_transform_em) instead of the Sun-Earth
one, and the two extra orientation parameters the Earth-Moon transform needs
(Omega_node, i_moon) are threaded through. source_def, data_read (Earth half)
are imported UNCHANGED. Earth-bound satellites use the identical Keplerian
path as the original -- nothing about them changes.

Changes vs uvw_eval.py, all marked  # EM:
  - import coordinate_transform_em as ct   (was coordinate_transform)
  - fourier_volume(): l2_Omega, l2_imoon args; passed to ct.l2_synodic_to_eci
  - uvw_cal_s(): l2_Omega_node, l2_i_moon args; threaded per L2 sat
  - output filename tagged _EM_ and uses the actual observation time
============================================================================
"""
import numpy as np
import csv
import pandas as pd
import pickle
import os
import h5py  # Used to work with HDF5 files for storing large amounts of data
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Suppress warnings for deprecated NumPy features
import argparse
import datetime
import astropy
from astropy.time import Time  # Handling astronomical time formats
from astropy.coordinates import SkyCoord, Angle  # Handling celestial coordinates
import pymap3d as pm  # Used for converting between coordinate systems (geodetic, ECEF, ENU)

# Import custom modules for source definitions, coordinate transformations, and data reading
import source_def as sf
from source_def import *
import coordinate_transform_em as ct          # EM: Earth-Moon CR3BP
from coordinate_transform_em import *          # EM
import er3bp_periodic_em as ep                 # EM: Earth-Moon ER3BP (periodic halo)
#import data_read as dr
#from data_read import *


## Constants
Me = 5.972e24  # Mass of Earth in kg
Mn = 7.34767309e22 # Mass of Moon in kg
G = 6.6743e-11  # Gravitational constant in SI units

######################## SVLBI FUNCTIONS ########################




##############ADDED###################################

def get_max_baseline_and_cell(d, freq_hz=55e6):
    """Extract GLOBAL max baseline from your d array → cell size"""
    
    all_lengths = []
    baseline_maxes = []
    
    for i, baseline_data in enumerate(d):
        bl_lengths = np.array(baseline_data).flatten()
        all_lengths.extend(bl_lengths)
        baseline_max = np.max(bl_lengths)
        baseline_maxes.append(baseline_max)
        print(f"Baseline {i+1}: Max = {baseline_max/1e3:.0f} km")
    
    # GLOBAL maximum
    B_max = np.max(all_lengths)
    print(f"\n GLOBAL B_MAX = {B_max/1e3:.0f} km")
    
    # Cell size calculation
    c = 3e8
    lam = c / freq_hz  # 6m @ 50 MHz
    B_max_lambda = B_max / lam  
    resolution_as = 206265 / B_max_lambda
    cell_as = resolution_as / 4  #sampling factor = 4
    
    print(f"Resolution: {resolution_as:.2f}\"")
    print(f"CASA cell size: '{cell_as:.2f}as'")
    
    return B_max, cell_as



# Function to calculate pairwise differences in a list
def diff(x):
    """
    Calculate pairwise differences between all elements in an array.

    Parameters:
    x (array-like): Input array

    Returns:
    array: Array of pairwise differences
    """
    x_diff = [(x[i] - x[j]) for i in range(len(x)) for j in range(i+1, len(x))]
    return np.asarray(x_diff)

# Function to calculate u, v, w coordinates in SVLBI mode
def fourier_volume(time, itime, A, E, tnm, W, OMG, I, rad, ra, dec, Mp = Me,
                   kind=None, l2_halo=None, l2_theta0=None,
                   l2_Omega=None, l2_imoon=None,
                   model='cr3bp', l2_orbit=None): # works for a pair   # EM/ER3BP
    """
    Calculate the u, v, w coordinates for SVLBI mode, for ONE baseline pair.

    Parameters:
    time (float): Total observation time
    itime (float): Integration time
    A, E (list): Semi-major axis and eccentricity for each satellite
    tnm (list): True anomaly for each satellite
    W, OMG, I (list): Argument of periapsis, right ascension of ascending node, and inclination for each satellite
    rad (list): Orbital period for each satellite
    ra, dec (float): Right ascension and declination of the source (in radians)

    L2-satellite extension (back-compatible: omit these and behaviour is unchanged):
    kind (list len 2): 'earth' or 'l2' for each member of the pair. Default: both 'earth'.
    l2_halo (list len 2): cached halo dict (from ct.build_l2_halo) for an 'l2' member, else None.
    l2_theta0 (list len 2): initial synodic->inertial angle theta_0 (rad) for an 'l2' member.

    Both satellites are placed in the SAME ECI frame each timestep, so the
    baseline R = R_i_ECI - R_j_ECI is valid whether a member is Earth-bound
    (peri2eci) or at L2 (synodic -> ECI per SVLBI_paper).

    Returns:
    tuple: u, v, w coordinates and baseline lengths
    """
    if kind is None:
        kind = ['earth', 'earth']            # default = original behaviour
    if l2_halo is None:
        l2_halo = [None, None]
    if l2_theta0 is None:
        l2_theta0 = [0.0, 0.0]
    if l2_Omega is None:                                   # EM
        l2_Omega = [np.deg2rad(340.575), np.deg2rad(340.575)]   # EM (2026 epoch default)
    if l2_imoon is None:                                   # EM
        l2_imoon = [np.deg2rad(5.145), np.deg2rad(5.145)]  # EM
    if l2_orbit is None:                                   # ER3BP
        l2_orbit = [None, None]                            # ER3BP

    Eci_x, Eci_y, Eci_z = [], [], []  # Earth-Centered Inertial (ECI) coordinates
    u, v, w = [], [], []  # Baseline coordinates in u, v, w space
    D = []  # Baseline distance

    T = 0  # Start observation at time T = 0
    while T < time:
        for k in range(2):  # Loop over both satellites
            if kind[k] == 'l2':
                # L2 satellite: sample the (cached) halo and transform
                # synodic -> ECI at THIS timestep T  [SVLBI_paper Eq.10,22-60].
                # The transform is recomputed every timestep because theta(t)
                # and the halo phase both advance with T.
                if model == 'er3bp':                                    # ER3BP
                    # Periodic ER3BP halo: physical time -> ECI via the
                    # pulsating-frame transform (Ferrari & Lavagna 2021).
                    x = ep.l2_er3bp_to_eci(l2_orbit[k], T, theta0=l2_theta0[k],
                                           Omega_node=l2_Omega[k], i_moon=l2_imoon[k])
                else:                                                   # CR3BP
                    x = ct.l2_synodic_to_eci(l2_halo[k], T, theta0=l2_theta0[k],
                                             Omega_node=l2_Omega[k], i_moon=l2_imoon[k])   # EM
            else:
                # Earth-bound satellite: Keplerian -> ECI (unchanged logic)
                e = E[k]
                f_ii = tnm[k]  # Initial true anomaly
                T_p_k   = (2 * np.pi) / np.sqrt((G * Me) / A[k]**3)   # Kepler formula
                Rad_S_k = (2 * np.pi) / T_p_k
                t_anom = ct.true_anom(A[k], e, f_ii, T, Rad_S_k, Mp)    # Calculate updated true anomaly
                x = ct.peri2eci(A[k], e, t_anom, W[k], OMG[k], I[k], T)  # Convert periapsis to ECI coordinates
            Eci_x.append(x[0])
            Eci_y.append(x[1])
            Eci_z.append(x[2])

        # Compute differences in ECI coordinates between satellite pairs (baselines)
        X = diff(Eci_x)
        Y = diff(Eci_y)
        Z = diff(Eci_z)
        d = np.sqrt(X**2 + Y**2 + Z**2)  # Baseline lengths
        D.append(d[:])

        # Calculate unit vectors u_cap, v_cap, w_cap based on source direction
        u_cap, v_cap, w_cap = sf.uvw_cap(ra, dec)
        for j in range(len(X)):  # Project the baseline onto u, v, w directions
            r = np.array([[X[j], Y[j], Z[j]]])
            u.append(np.dot(u_cap, r.T))
            v.append(np.dot(v_cap, r.T))
            w.append(np.dot(w_cap, r.T))
    
        # Clear the lists for next iteration and update the time
        Eci_x.clear()
        Eci_y.clear()
        Eci_z.clear()
        T += itime  # Increment time by integration time

    return u, v, w, D

# Function to calculate u, v, w coordinates for multiple phase-centers
def uvw_cal_s(time, itime, RA, DEC, A, E, I, OMG, W, T_Anm, T_p, Rad_S,
              l2_Az_km=None, l2_type=None, l2_phase0=None, l2_theta0=None,
              l2_Omega_node=None, l2_i_moon=None,
              model='cr3bp', er3bp_e=0.0554):               # EM/ER3BP
    """
    Calculate u, v, w coordinates for multiple phase centers over the observation.

    Parameters:
    time (float): Total observation time
    itime (float): Integration time
    RA, DEC (float): Right ascension and declination of the source
    A, E, I, OMG, W (list): Orbital parameters for the EARTH-BOUND satellites
    T_Anm (list): True anomaly
    T_p (float): Time period of the orbit
    Rad_S (list): Orbital period for the satellites

    L2-satellite extension (back-compatible: omit these and behaviour is unchanged):
    l2_Az_km, l2_type, l2_phase0, l2_theta0 (arrays): one entry per L2 satellite,
        as read by dr.l2_sat_data(). Each L2 satellite is appended to the list
        of stations, so it joins EVERY pair (L2<->Earth-sat and L2<->L2).

    Returns:
    list: Baseline lengths
    """
    print('s',RA)
    ra = sf.ra_conversion(RA)  # Convert RA from hours to radians
    dec = sf.dec_conversion(DEC)  # Convert DEC from degrees to radians

    # ---- Build a unified station list: Earth-bound sats first, then L2 sats ----
    # 'kind' tags each station so fourier_volume knows how to propagate it.
    n_earth = len(A)
    A_all   = list(A);   E_all = list(E);     I_all = list(I)
    OMG_all = list(OMG); W_all = list(W);     TA_all = list(T_Anm); Rad_all = list(Rad_S)
    kind_all   = ['earth'] * n_earth
    halo_all   = [None] * n_earth
    orbit_all  = [None] * n_earth                          # ER3BP
    theta0_all = [0.0]  * n_earth
    Omega_all  = [0.0]  * n_earth                          # EM
    imoon_all  = [0.0]  * n_earth                          # EM

    # ER3BP: build the periodic double-revolution halo ONCE (cached seed).
    er3bp_orbit = None
    if model == 'er3bp' and l2_Az_km is not None and len(l2_Az_km) > 0:
        print(f"  [ER3BP] building periodic double-revolution L2 halo "
              f"(Ferrari & Lavagna 2021), e={er3bp_e} ...")
        er3bp_orbit = ep.build_er3bp_halo(e=er3bp_e, verbose=True)
        print(f"  [ER3BP] periodic halo ready (period 2*pi in true anomaly, "
              f"double-revolution). NOTE: the ER3BP halo amplitude is FIXED by "
              f"the 2:1 resonance; l2_Az_km is ignored in ER3BP mode.")

    if l2_Az_km is not None and len(l2_Az_km) > 0:
        for li in range(len(l2_Az_km)):
            # Build the CR3BP halo ONCE here (cached inside ct); the per-timestep
            # transform inside fourier_volume is the only repeated work.
            halo = ct.build_l2_halo(float(l2_Az_km[li]),
                                    int(l2_type[li]) if l2_type is not None else 1,
                                    float(l2_phase0[li]) if l2_phase0 is not None else 0.0)
            print(f"  L2 sat {li+1}: Az={halo['Az_km']:.0f} km, Ax={halo['Ax_km']:.0f} km, "
                  f"period={halo['T_full']* ct._TSTAR_S/86400:.1f} d")
            # placeholders for the Keplerian slots (unused for L2 members)
            A_all.append(0.0); E_all.append(0.0); I_all.append(0.0)
            OMG_all.append(0.0); W_all.append(0.0); TA_all.append(0.0); Rad_all.append(0.0)
            kind_all.append('l2')
            halo_all.append(halo)
            orbit_all.append(er3bp_orbit)                      # ER3BP (same for all L2 sats)
            theta0_all.append(float(l2_theta0[li]) if l2_theta0 is not None else 0.0)
            Omega_all.append(float(l2_Omega_node[li]) if l2_Omega_node is not None   # EM
                             else np.deg2rad(340.575))                                # EM
            imoon_all.append(float(l2_i_moon[li]) if l2_i_moon is not None            # EM
                             else np.deg2rad(5.145))                                  # EM

    d = []  # List to store baseline lengths
    mm = len(A_all)  # Total number of stations (Earth-bound + L2)

    _tag = 'ER3BP' if model=='er3bp' else 'CR3BP'                 # EM
    filename = f'uvw_{RA}_{DEC}_6I_EM_{_tag}_L2_{int(time)}.h5'   # EM

    print(RA)
    

    with h5py.File(filename, 'a') as hdf:
        for i in range(mm):
            for j in range(i + 1, mm):  # Loop through ALL station pairs (now incl. L2)
                # Prepare satellite orbital parameters for the current pair
                p = [A_all[i], A_all[j]]
                # print(p)
                q = [E_all[i], E_all[j]]
                r = [TA_all[i], TA_all[j]]
                # print(r)
                s = [W_all[i], W_all[j]]
                tk = [OMG_all[i], OMG_all[j]]
                gh = [I_all[i], I_all[j]]
                rad = [Rad_all[i], Rad_all[j]]
                kind = [kind_all[i], kind_all[j]]        # NEW
                l2h  = [halo_all[i], halo_all[j]]        # NEW
                th0  = [theta0_all[i], theta0_all[j]]    # NEW
                om0  = [Omega_all[i], Omega_all[j]]      # EM
                im0  = [imoon_all[i], imoon_all[j]]      # EM
                orb0 = [orbit_all[i], orbit_all[j]]      # ER3BP

                # Calculate u, v, w coordinates (pair may contain Earth and/or L2 sats)
                data = fourier_volume(time, itime, p, q, r, s, tk, gh, rad, ra, dec,
                                      Mp = Me, kind=kind, l2_halo=l2h, l2_theta0=th0,
                                      l2_Omega=om0, l2_imoon=im0,
                                      model=model, l2_orbit=orb0)          # EM/ER3BP

                # Label the pair: L# for an L2 station, plain number for Earth-bound
                def _lbl(idx):
                    return f"L{idx-n_earth+1}" if kind_all[idx]=='l2' else f"{idx+1}"
                tag = f"{_lbl(i)}{_lbl(j)}"

                # Save u, v, w coordinates to HDF5 file
                grp = hdf.create_group(f'Sat_{tag}')
                grp.create_dataset(f"u_{tag}_{RA}_{DEC}", data=np.asarray(data[0]))
                grp.create_dataset(f"v_{tag}_{RA}_{DEC}", data=np.asarray(data[1]))
                grp.create_dataset(f"w_{tag}_{RA}_{DEC}", data=np.asarray(data[2]))
                d.append(data[3])
   
    # USAGE (after running uvw_cal_s) ADDED ###########
    B_max, cell = get_max_baseline_and_cell(d)
    print("MAXIMUM BASELINE LENGTH =",B_max)
    print("CellSize for tclean =" , cell)

    return d  # Return baseline lengths



# Uncomment this block if you want to run this script directly and pass filenames from the command line
# if __name__ == '__main__' :
    
#     parser = argparse.ArgumentParser(description='provide the observation time, integration time')
#     parser.add_argument('mode',type=str, help='Mode of operation\n s-SVLBI \n g -GVLBI \n s+g-both modes')
#     parser.add_argument('time',type=float)
#     parser.add_argument('int_time',type=float)
#     parser.add_argument('index_of_reference_latitude',type=int)
#     parser.add_argument('index_of_reference_longitude',type=int)
#     parser.add_argument('index_of_reference_height',type=int)
    
#     arg = parser.parse_args()
    
#     time = arg.time
#     itime = arg.int_time  ## integration time
#     RA = sf.ra
#     DEC = sf.dec
#     ant_ref_data = dr.ant_data
    
#     r_lat = ant_ref_data[0][arg.index_of_reference_latitude]
#     r_long = ant_ref_data[1][arg.index_of_reference_longitude]
#     r_ht = ant_ref_data[2][arg.index_of_reference_height]

    
#     # Run the functions with the provided filenames
#     if args.mode == 's':
#         fourier_volume(A,E,I,OMG,W,T_Anm,T_p,Rad_S,time,itime,RA,DEC)
#     if args.mode == 'g':    
#         uvw_cal_g(time,itime,r_long,r_lat,r_ht,RA,DEC)
#     if args.mode == 's+g':
#         fourier_volume(time,itime,RA,DEC)
#         uvw_cal_g(time,itime,r_long,r_lat,r_ht,RA,DEC)
#         uvw_cal_sg(time,itime,r_long,r_lat,r_ht,RA,DEC)

########## elevation limit ta add korte hobe ################ 