"""
data_read_em.py
============================================================================
EARTH-MOON replacement for the L2 reader in data_read.py.

ONLY l2_sat_data() changes. sat_data() and ant_data() are UNCHANGED, because
Earth-bound satellites are still described by Keplerian elements and ground
antennas by geodetic coordinates regardless of which CR3BP is used. Import
those two from the original data_read module.

WHAT CHANGED: the Earth-Moon synodic plane is the LUNAR ORBIT PLANE rather
than the ecliptic, so its orientation relative to the ecliptic must be
supplied. Two columns are therefore added to the L2 parameter file:
    Omega_node_deg  -- longitude of the lunar ascending node on the ecliptic
    i_moon_deg      -- inclination of the lunar orbit to the ecliptic
Both are consumed by coordinate_transform_em.l2_synodic_to_eci() Stage 3.

Backwards compatible: if a 4-column (old Sun-Earth style) file is supplied,
the two new columns default to the J2000 mean values, and a warning is
printed so the default is never silently assumed.
============================================================================
"""

import numpy as np
import pandas as pd

# Defaults if the file predates the two new columns.
#   Omega at J2000.0 = 125.0445479 deg  [Meeus 1998, Astron. Algorithms, Ch.47 Eq.47.7]
#   mean lunar inclination to ecliptic  = 5.145 deg  [Meeus 1998 Ch.47]
_OMEGA_NODE_J2000_DEG = 125.0445479
_I_MOON_MEAN_DEG = 5.145

# Adjust to wherever your input files live (kept identical to the original).
_INPUT_DIR = '/home/tiasha/SVLBI_FOLDER/aisim/simulation_input/'


def sat_data(filename1):  ## Accepts a filename as input
    # Reading the CSV file containing satellite data
    file = '/home/tiasha/SVLBI_FOLDER/aisim/simulation_input/'+filename1  # Path to the satellite data file
    Data = pd.read_csv(file, delimiter=",")  # Reads the CSV file using pandas, with a comma delimiter
    # Print the satellite data
    print('Satellite data:\n', Data)
    
    # Converting the DataFrame to a numpy array for easier manipulation
    data = Data.to_numpy()
    
    # Initializing arrays to store various satellite parameters
    A = np.zeros(len(data))        # Semi-major axes
    E = np.zeros(len(data))        # Eccentricities
    I = np.zeros(len(data))        # Inclinations
    OMG = np.zeros(len(data))      # Longitude of ascending node
    W = np.zeros(len(data))        # Argument of periapsis
    T_p = np.zeros(len(data))      # Time periods (non-zero for elliptical orbits)
    T_Anm = np.zeros(len(data))    # Initial true anomalies
    
    # Iterating over the data to extract and convert each satellite parameter
    for i in range(len(data)):
        A[i] = data[i][0]              # Semi-major axis
        E[i] = data[i][1]              # Eccentricity
        I[i] = np.deg2rad(data[i][2])  # Inclination converted from degrees to radians
        OMG[i] = np.deg2rad(data[i][3])# Longitude of ascending node in radians
        W[i] = np.deg2rad(data[i][4])  # Argument of periapsis in radians
        T_Anm[i] = np.deg2rad(data[i][5]) # Initial true anomaly in radians
        #T_p[i] = data[i][6]            # Time period of satellite (for circular orbits) ORIGINALLLLLLLLLLLLLLLLLLLLLLLLLLL

        G = 6.6743e-11
        Me = 5.972e24
        T_p[i] = 2 * np.pi * np.sqrt(data[i][0]**3 / (G * Me))







    # Calculating the radial speed for circular orbits (angular speed = 2*pi / time period)
    Rad_S = (2 * np.pi) / T_p
    print(T_Anm)
    # Returning all the parameters as numpy arrays
    return A, E, I, OMG, W, T_Anm, T_p, Rad_S

def l2_sat_data_em(filename_l2, input_dir=_INPUT_DIR):
    """Read the EARTH-MOON L2-satellite parameter file.

    Expected columns (comma separated, '#' comments allowed):
        Az_km, halo_type, phase0_deg, theta0_deg,
        Omega_node_deg, i_moon_deg [, eps_band, N_rev]

    Returns six numpy arrays (one entry per L2 satellite):
        Az_km, halo_type, phase0_rad, theta0_rad, Omega_node_rad, i_moon_rad

    NOTE the Az_km scale change: for Earth-Moon the natural amplitude unit is
    gamma*L* = 64,573 km (vs 1,501,520 km for Sun-Earth), so sensible inputs
    are ~1,000-30,000 km, NOT the old 160,000 km. A loud warning is printed if
    an obviously Sun-Earth-scale amplitude is supplied.
    """
    path = input_dir + filename_l2
    Data = pd.read_csv(path, delimiter=",", comment='#', header=None)
    print('L2 satellite data (Earth-Moon):\n', Data)

    data = Data.to_numpy()
    n = len(data)
    ncol = data.shape[1]

    Az_km     = np.zeros(n)
    halo_type = np.zeros(n)
    phase0    = np.zeros(n)
    theta0    = np.zeros(n)
    Omega_nd  = np.zeros(n)
    i_moon    = np.zeros(n)

    if ncol < 6:
        print(f"[data_read_em] WARNING: file has only {ncol} columns; this looks "
              f"like an old Sun-Earth L2 file. Defaulting Omega_node="
              f"{_OMEGA_NODE_J2000_DEG} deg and i_moon={_I_MOON_MEAN_DEG} deg "
              f"(J2000 mean values). Supply them explicitly for a real epoch.")

    for i in range(n):
        Az_km[i]     = float(data[i][0])
        halo_type[i] = int(data[i][1])
        phase0[i]    = np.deg2rad(float(data[i][2]))
        theta0[i]    = np.deg2rad(float(data[i][3]))
        Omega_nd[i]  = np.deg2rad(float(data[i][4]) if ncol > 4 else _OMEGA_NODE_J2000_DEG)
        i_moon[i]    = np.deg2rad(float(data[i][5]) if ncol > 5 else _I_MOON_MEAN_DEG)

        if Az_km[i] > 60000.0:
            print(f"[data_read_em] WARNING: Az_km={Az_km[i]:,.0f} for satellite "
                  f"{i+1} exceeds the Earth-Moon L2 scale (gamma*L* = 64,573 km). "
                  f"This looks like a leftover Sun-Earth amplitude. Expected "
                  f"range for Earth-Moon is ~1,000-30,000 km.")

    return Az_km, halo_type, phase0, theta0, Omega_nd, i_moon


if __name__ == '__main__':
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else 'l2_sat_em.txt'
    d = sys.argv[2] if len(sys.argv) > 2 else './'
    out = l2_sat_data_em(fn, input_dir=d)
    names = ['Az_km', 'halo_type', 'phase0_rad', 'theta0_rad', 'Omega_node_rad', 'i_moon_rad']
    for nm, arr in zip(names, out):
        print(f"  {nm:16s} = {arr}")
