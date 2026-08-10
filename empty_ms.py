import numpy as np
import os
import sys
import shutil
import time
from os import path
from casacore.tables import table, maketabdesc, makearrcoldesc
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys
from casatasks.private import simutil

# ============================================================================
#  EMPTY MEASUREMENT-SET (MS) CREATION  --  SATELLITE (s) MODE, L2 pipeline
# ----------------------------------------------------------------------------
#  This version is for SATELLITE mode only. There are NO ground antennas, so
#  there is NO station/antenna file input.
#
#  Why we still "make an array":
#    CASA needs to know HOW MANY stations exist, because that fixes the number
#    of baselines  C(n,2)  and therefore the number of MS rows
#    ( = n_baselines * n_timesteps ). That row count is baked into the MS by
#    sm.observe() and is NOT overwritten later.
#
#  Why the positions are dummies:
#    In satellite mode the real UVW values come from the pipeline HDF5 and are
#    written into the MS afterwards (by fill_uvw_predict.py), OVERWRITING
#    whatever UVW CASA computes here. So the x/y/z handed to setconfig only
#    need to (a) be the right COUNT and (b) be distinct (no zero-length
#    baselines). Their actual values are irrelevant.
#
#  L2: the L2 satellite is simply one more station. For 5 Earth sats + 1 L2,
#  pass  n_sat = 6  ->  C(6,2) = 15 baselines.
# ============================================================================

start = time.time()

# Instantiate the required CASA tools
sm   = simulator()
ia   = image()
tb   = table()
cs   = coordsys()
me   = measures()
qa   = quanta()
cl   = componentlist()
mysu = simutil.simutil()


def make_dummy_array(n_sat):
    """Build a dummy (n_sat x 4) [diam, x, y, z] array for satellite mode.

    Positions are placeholders (they get overwritten by the pipeline UVW); they
    only need to be distinct so CASA does not see zero-length/duplicate
    baselines. The COUNT is what matters -> it sets C(n_sat, 2) baselines.
    """
    global stna
    stna = np.zeros([n_sat, 4], float)
    stna[:, 0] = 5.0                      # dish diameter (arbitrary, e.g. 5 m)
    stna[:, 1] = np.arange(n_sat) * 1.0   # x: 0,1,2,... (distinct)
    stna[:, 2] = np.zeros(n_sat)          # y
    stna[:, 3] = np.zeros(n_sat)          # z

    n_baselines = n_sat * (n_sat - 1) // 2
    print(f"[array] stations (satellites) = {n_sat}")
    print(f"[array] baselines C(n,2)      = {n_baselines}")
    return stna, n_sat


def simulate(msname, direction, spwname, freq, deltaf, freqres, nchan, stokes,
             clname, itime, stime, endtime):
    """Create and dimension the empty MS for the satellite array and timing."""
    sname = 'src_1'

    sm.open(msname)
    dir0 = f'{direction}'

    postel = me.observatory('MWA32T')
    sm.setconfig(telescopename='VLA',
                 x=stna[:, 1], y=stna[:, 2], z=stna[:, 3],
                 dishdiameter=stna[:, 0], antname='SAT',
                 coordsystem='local', referencelocation=postel)

    sm.setspwindow(spwname=f'{spwname}', freq=f'{freq}', deltafreq=f'{deltaf}',
                   freqresolution=f'{freqres}', nchannels=int(nchan),
                   stokes=f'{stokes}')
    sm.setfeed(mode='perfect X Y', pol=['R'])
    sm.setfield(sourcename=sname, sourcedirection=dir0)
    sm.setauto(0.0)
    sm.settimes(integrationtime=f'{itime}s', usehourangle=False,
                referencetime=me.epoch('UTC', 'today'))
    sm.observe(sourcename=sname, spwname=f'{spwname}',
               starttime=qa.quantity(stime, "s"),
               stoptime=qa.quantity(endtime, "s"))
    sm.close()
    sm.done()


import argparse

parser = argparse.ArgumentParser(description='Create an empty MS (satellite mode, L2 pipeline)')
parser.add_argument('msfile', type=str, help='Name for the empty MS file')
parser.add_argument('n_sat', type=int, help='Number of satellites (e.g. 5 Earth + 1 L2 = 6)')
parser.add_argument('endtime', type=int, help='Total observation time (sec)')
parser.add_argument('int_time', type=int, help='Integration time (sec)')
parser.add_argument('Band_name', type=str, help='Name of band, e.g. band0')
parser.add_argument('Direction', type=str, help='Phasecenter, e.g. "J2000 00h00m00 00d00m00"')
parser.add_argument('Frequency', type=str, help='Starting frequency, e.g. 50MHz')
parser.add_argument('Frequency_resolution', type=str, help='Freq resolution, e.g. 10kHz') # Put 10MHz itself if ur taking number of chnnels = 1 because BW = 10MHz
parser.add_argument('Bandwidth', type=str, help='Bandwidth, e.g. 10MHz')
parser.add_argument('N_chan', type=int, help='Number of channels')
parser.add_argument('Stokes_Parameters', type=str, help='Stokes, e.g. "RR" or "LL"')

args = parser.parse_args()

# ---- inputs (use argparse values directly) ----
msname   = args.msfile
clname   = '/home/tiasha/SVLBI_ORIGINALL/Pointsrc_20Aug.cl'   # sky-model component list

stime    = 0.0
endtime  = float(args.endtime)
itime    = float(args.int_time)

spwname   = args.Band_name
freq      = args.Frequency
deltaf    = args.Bandwidth
freqres   = args.Frequency_resolution
nchan     = args.N_chan
stokes    = args.Stokes_Parameters
direction = args.Direction

# ---- build the dummy satellite array (count = n_sat, incl. the L2 sat) ----
make_dummy_array(args.n_sat)

# expected MS rows = n_baselines * n_timesteps  (cross-check)
n_baselines = args.n_sat * (args.n_sat - 1) // 2
n_timesteps = int(round(endtime / itime))
print(f"[expected] timesteps = {n_timesteps}, "
      f"MS rows = baselines*timesteps = {n_baselines} * {n_timesteps} "
      f"= {n_baselines * n_timesteps}")

simulate(msname, direction, spwname, freq, deltaf, freqres, nchan, stokes,
         clname, itime, stime, endtime)

print('\n\nEmpty MS file done\n')
print(f"Total time: {time.time()-start:.1f}s")