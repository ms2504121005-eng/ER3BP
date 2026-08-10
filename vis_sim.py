import numpy as np
import os
import sys
import shutil
import time
import h5py
from casacore.tables import table as cctable
from casatools import simulator, image, table, coordsys, measures, componentlist, quanta, ctsys
from casatasks import flagdata
from casatasks.private import simutil

# ============================================================================
#  FILL UVW FROM HDF5  +  PREDICT VISIBILITIES  --  adapted for the L2 pipeline
# ----------------------------------------------------------------------------
#  What changed vs the old version, and WHY:
#  (1) L2 ADAPTATION: the old code assumed 10 baselines x 10000 timesteps =
#      100000 rows. With the L2 satellite the number of baselines changes
#      (e.g. 5 Earth + 1 L2 -> C(6,2) = 15). We now read the number of
#      baselines and timesteps DIRECTLY from the HDF5 file and shape the UVW
#      block from those, so it works for ANY satellite count.
#  (2) SAFETY: before writing, we CHECK that (n_baselines * n_timesteps) equals
#      the number of rows in the MS. If they don't match, we STOP with a clear
#      message instead of silently writing a misaligned UVW column (which would
#      corrupt every visibility). This is the single most important guard here.
#  (3) The visibility prediction itself is UNCHANGED: a baseline is a baseline
#      whether or not it involves the L2 satellite, so predictSim needs no edit.
#
#  HONEST CAVEAT: the interleaving order below (timestep-major: for each
#  timestep, all baselines) MUST match the row ordering that CASA's sm.observe
#  produced in the empty MS. The old code used transpose(1,0,2); we keep that
#  but VERIFY the total row count. If your CASA build orders rows differently
#  (baseline-major), the totals still match but the per-row UVW would be
#  permuted -- see the note at the write step and verify on a small case.
# ============================================================================

import argparse as _argparse
_parser = _argparse.ArgumentParser(description='Fill UVW from HDF5 and predict visibilities (L2 pipeline)')
_parser.add_argument('msname', type=str, help='Path to the empty MS file (created by empty_ms.py)')
_parser.add_argument('file_path', type=str, help='Path to the pipeline HDF5 UVW file')
_parser.add_argument('usemod', type=str, choices=['cl', 'im'], help="Sky model type: 'cl' or 'im'")
_parser.add_argument('model_path', type=str, help='Path to .cl (usemod=cl) or .im (usemod=im)')
_args = _parser.parse_args()

msname    = _args.msname
file_path = _args.file_path
usemod    = _args.usemod
clname    = _args.model_path if usemod == 'cl' else ''
imname    = _args.model_path if usemod == 'im' else ''

# -- instantiate CASA tools ---------------------------------------------------
start = time.time()
sm   = simulator()
ia   = image()
tb   = table()
cs   = coordsys()
me   = measures()
qa   = quanta()
cl   = componentlist()
mysu = simutil.simutil()


def extract_uvw_from_group(group):
    """Extract (u, v, w) from one HDF5 baseline group -> array (n_timesteps, 3)."""
    u_key = next((k for k in group.keys() if k.startswith("u_")), None)
    v_key = next((k for k in group.keys() if k.startswith("v_")), None)
    w_key = next((k for k in group.keys() if k.startswith("w_")), None)
    if u_key and v_key and w_key:
        u = np.squeeze(group[u_key][()])
        v = np.squeeze(group[v_key][()])
        w = np.squeeze(group[w_key][()])
        return np.stack([u, v, w], axis=1)        # (n_timesteps, 3)
    return None


def predictSim(msname, imname, clname, usemod, usepb):
    """Predict visibilities into the DATA column (unchanged by L2)."""
    sm.setlimits(shadowlimit=1.0, elevationlimit='-90deg')   # ignore horizon/shadowing
    sm.openfromms(msname)
    if usepb:
        sm.setvp(dovp=True, usedefaultvp=True)
    if usemod == 'im':
        sm.predict(imagename=imname, incremental=False)
    if usemod == 'cl':
        sm.predict(complist=clname, incremental=False)
    sm.close()


# ── load UVW from HDF5 (auto-sized: works for any number of baselines) ───────
print("Loading HDF5...")
all_uvw    = []
final_data = {}

with h5py.File(file_path, "r") as hdf5_file:
    # sorted() gives a deterministic baseline order; keep it consistent with
    # how the empty MS was built.
    for group_name in sorted(hdf5_file.keys()):
        group = hdf5_file[group_name]
        if isinstance(group, h5py.Group):
            uvw = extract_uvw_from_group(group)
            if uvw is not None:
                final_data[group_name] = uvw
                all_uvw.append(uvw)

n_baselines = len(all_uvw)                       # e.g. 15 for 6 satellites
if n_baselines == 0:
    sys.exit("ERROR: no UVW groups found in the HDF5 file. Check file_path / group names.")

n_timesteps = all_uvw[0].shape[0]                # e.g. 288 for 2 days @ 600 s

# sanity: every baseline must have the same number of timesteps
for i, arr in enumerate(all_uvw):
    if arr.shape[0] != n_timesteps:
        sys.exit(f"ERROR: baseline {i} has {arr.shape[0]} timesteps but "
                 f"baseline 0 has {n_timesteps}. HDF5 is inconsistent.")

print(f"Baselines: {n_baselines}, Timesteps: {n_timesteps}")
print("Baseline groups (order used):")
for k in sorted(final_data.keys()):
    print(f"    {k}")

# stack -> (n_baselines, n_timesteps, 3)
uvw_array = np.array(all_uvw)

# Interleave to MS row order: timestep-major (for each timestep, all baselines)
# (n_baselines, n_timesteps, 3) -> (n_timesteps, n_baselines, 3) -> (rows, 3)
uvw_interleaved = uvw_array.transpose(1, 0, 2).reshape(-1, 3)
expected_rows   = n_baselines * n_timesteps
print(f"Final UVW shape: {uvw_interleaved.shape}  (expected rows = {expected_rows})")

# ── write UVW into MS, WITH a hard row-count check (the key L2 safety) ───────
print("Opening MS to write UVW...")
tbl2 = cctable(msname, readonly=False)
ms_rows = tbl2.nrows()
print("MS rows:", ms_rows)

if ms_rows != expected_rows:
    tbl2.close()
    sys.exit(
        "ERROR: row-count mismatch.\n"
        f"  HDF5 implies {n_baselines} baselines x {n_timesteps} timesteps "
        f"= {expected_rows} rows,\n"
        f"  but the MS has {ms_rows} rows.\n"
        "  => The empty MS was built with a DIFFERENT number of stations or "
        "timesteps than the HDF5.\n"
        "     Rebuild the empty MS with n_ant = number of satellites "
        "(Earth + L2) and the SAME endtime/int_time used for the pipeline run."
    )

# CASA's UVW column shape is (nrows, 3); our array matches.
tbl2.putcol("UVW", uvw_interleaved)
tbl2.close()
print("UVW written successfully")

# ── unflag all data ──────────────────────────────────────────────────────────
flagdata(vis=msname, mode='unflag')
print("Data unflagged")

# ── predict visibilities (unchanged by L2) ───────────────────────────────────
print("Running predictSim...")
predictSim(msname=msname,
           imname=imname,
           clname=clname,
           usemod=usemod,
           usepb=False)

print('\nDone! Visibilities stored in DATA column.')
print(f"Total time: {time.time()-start:.1f}s")
