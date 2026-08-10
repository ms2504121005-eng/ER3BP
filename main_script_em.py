"""
main_script_em.py  --  EARTH-MOON variant of main_script.py.
============================================================================
Identical to main_script.py EXCEPT it drives the Earth-Moon pipeline:
  - uvw_eval_em   (Earth-Moon L2 propagation)  instead of uvw_eval
  - data_read_em.l2_sat_data_em (6-column L2 file) for the L2 params, so the
    two extra Earth-Moon orientation columns (Omega_node, i_moon) are read
    and threaded to uvw_cal_s.
Earth-bound satellites are still read by the UNCHANGED data_read.sat_data.
source_def is unchanged. Only process_satellite is shown; the antenna/combined
paths are Earth-only and irrelevant to the L2 study, so they are omitted.
============================================================================
"""
import argparse
import subprocess


import source_def as sf
from source_def import *
import coordinate_transform_em as ct          # EM
from coordinate_transform_em import *          # EM
#import data_read as dr                          # Earth sats: UNCHANGED reader
#from data_read import *
import data_read_em as drem                      # EM: L2 sats (6-column file)
import uvw_eval_em as cal                        # EM
from uvw_eval_em import * 

## Constants
Me = 5.972e24  # Mass of Earth in kg
Mn = 7.34767309e22 # Mass of Moon in kg
G = 6.6743e-11  # Gravitational constant in SI units

def process_satellite(file_sat, time, itime, RA, DEC, file_l2=None,
                      model='cr3bp', er3bp_e=0.0554):        # ER3BP
    A, E, I, OMG, W, T_Anm, T_p, Rad_S = drem.sat_data(file_sat)
    # print('z',T_Anm)
    # print(RA)
    # If an L2-satellite file is provided, read it and pass it through so the
    # L2 satellite(s) join every baseline pair. Omitting file_l2 keeps the
    # original Earth-only behaviour unchanged.
    if file_l2:
        # EM: 6-column L2 file -> Az, type, phase0, theta0, Omega_node, i_moon
        (l2_Az_km, l2_type, l2_phase0, l2_theta0,
         l2_Omega, l2_imoon) = drem.l2_sat_data_em(file_l2)
        cal.uvw_cal_s(time, itime, RA, DEC, A, E, I, OMG, W, T_Anm, T_p, Rad_S,
                      l2_Az_km=l2_Az_km, l2_type=l2_type,
                      l2_phase0=l2_phase0, l2_theta0=l2_theta0,
                      l2_Omega_node=l2_Omega, l2_i_moon=l2_imoon,
                      model=model, er3bp_e=er3bp_e)                    # EM/ER3BP
    else:
        cal.uvw_cal_s(time, itime, RA, DEC, A, E, I, OMG, W, T_Anm, T_p, Rad_S)
    

def process_antenna(file_ant, time, itime, RA, DEC, index_lat, index_long, index_ht):
    lat, longi, ht = dr.ant_data(file_ant)
    r_lat = lat[index_lat]
    r_long = longi[index_long]
    r_ht = ht[index_ht]
    cal.uvw_cal_g(lat, longi, ht, time, itime, r_long, r_lat, r_ht, RA, DEC)
    return r_lat, r_long, r_ht

def process_combined(file_sat, file_ant, time, itime, RA, DEC, index_lat, index_long, index_ht ,Mp):
    # Process satellite and antenna data
    A, E, I, OMG, W, T_Anm, T_p, Rad_S = dr.sat_data(file_sat)
    lat, longi, ht = dr.ant_data(file_ant)
    r_lat = lat[index_lat]
    r_long = longi[index_long]
    r_ht = ht[index_ht]

    # Run all calculations
    cal.uvw_cal_s(time, itime, RA, DEC,A, E, I, OMG, W, T_Anm, T_p, Rad_S, ct.Me)
    cal.uvw_cal_g(lat, longi, ht, time, itime, r_long, r_lat, r_ht, RA, DEC)
    cal.uvw_cal_sg(A, E, I, OMG, W, T_Anm, T_p, Rad_S, lat, longi, ht, time, itime, r_long, r_lat, r_ht, RA, DEC, ct.Me)

if __name__ == "__main__":
    # Argument parsing setup
    parser = argparse.ArgumentParser(description="Wrapper script for processing satellite and antenna data.")
    subparsers = parser.add_subparsers(title='Utilities', description='Available utilities', dest='mode')

    # Common arguments
    parser.add_argument('time', type=int, help='Observation time.')
    parser.add_argument('int_time', type=int, help='Integration time.')
    parser.add_argument('RA', type=str, help='Right Ascension (h:m:s) format.')
    parser.add_argument('DEC', type=str, help='Declination (d:m:s) format.')

    # Satellite utility arguments
    parser_sat = subparsers.add_parser('s', help='Satellite processing.')
    parser_sat.add_argument('satellite_file', type=str, help='Satellite data file (CSV/Txt).')
    parser_sat.add_argument('l2_file', nargs='?', default=None, type=str,
                            help='Optional L2-satellite parameter file (e.g. l2_sat.txt).')

    # Antenna utility arguments
    parser_ant = subparsers.add_parser('g', help='Antenna processing.')
    parser_ant.add_argument('antenna_file', type=str, help='Antenna data file (CSV/Txt).')
    parser_ant.add_argument('index_of_reference_latitude', type=int, help='Reference latitude index.')
    parser_ant.add_argument('index_of_reference_longitude', type=int, help='Reference longitude index.')
    parser_ant.add_argument('index_of_reference_height', type=int, help='Reference height index.')

    # Combined utility arguments
    parser_combined = subparsers.add_parser('s+g', help='Combined satellite and antenna processing.')
    parser_combined.add_argument('satellite_file', type=str, help='Satellite data file (CSV/Txt).')
    parser_combined.add_argument('antenna_file', type=str, help='Antenna data file (CSV/Txt).')
    parser_combined.add_argument('index_of_reference_latitude', type=int, help='Reference latitude index.')
    parser_combined.add_argument('index_of_reference_longitude', type=int, help='Reference longitude index.')
    parser_combined.add_argument('index_of_reference_height', type=int, help='Reference height index.')

    # Parse arguments
    args = parser.parse_args()
    
    ra = args.RA
    dec = args.DEC

    # Run the appropriate function based on the mode
    if args.mode == 's':
        process_satellite(args.satellite_file, args.time, args.int_time, ra, dec,
                          file_l2=getattr(args, 'l2_file', None))
    elif args.mode == 'g':
        process_antenna(args.antenna_file, args.time, args.int_time,ra,dec,
                        args.index_of_reference_latitude, args.index_of_reference_longitude, args.index_of_reference_height)
    elif args.mode == 's+g':
        process_combined(args.satellite_file, args.antenna_file, args.time, args.int_time,ra,dec,
                         args.index_of_reference_latitude, args.index_of_reference_longitude, args.index_of_reference_height)
    else:
        parser.print_help()
    