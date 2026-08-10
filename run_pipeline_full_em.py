"""
run_pipeline_full_em.py  --  EARTH-MOON pipeline runner
============================================================================
Combined single-input wrapper for the SVLBI L2 pipeline, covering EVERY
stage from raw satellite/L2 files through to a calibrated MS ready for
tclean -- i.e. data_read -> source_def -> coordinate_transform -> uvw_eval
-> main_script -> plots -> empty_ms -> vis_sim.

WHAT THIS DELIBERATELY STOPS SHORT OF, per explicit instruction:
  tclean itself. tclean is run by hand inside CASA's own interactive shell
  (typing `casa`, then `tclean(...)` at the CASA<1> prompt) -- a separate
  process from system python3, not an importable/subprocess-able script.
  This wrapper produces the finished, UVW-filled, visibility-predicted MS;
  you then run tclean on it yourself, exactly as you do today.

WHAT THIS DOES NOT MODIFY:
  - data_read.py, source_def.py, coordinate_transform.py, uvw_eval.py,
    main_script.py, plots.py, empty_ms.py: byte-for-byte unmodified,
    called exactly as they exist in the project (imported where they have
    an __main__ guard / are import-safe; run as a subprocess with their
    OWN existing argparse where they are not, i.e. empty_ms.py).
  - No internal hardcoded path inside those files is changed (e.g.
    data_read.py's /home/tiasha/SVLBI_FOLDER/... input dir, or plots.py's
    /home/tiasha/SVLBI_L2_new/ HDF5-read prefix).

EARTH-MOON DIFFERENCE FROM run_pipeline_full.py:
  Only the dynamics modules differ. This wrapper imports main_script_em
  (which uses uvw_eval_em + coordinate_transform_em + data_read_em) instead
  of main_script. The CASA stages -- empty_ms.py and vis_sim.py -- are the
  SAME frame-agnostic files, called unchanged: they only build the MS shell
  and fill its UVW column from whatever HDF5 was written, and neither knows
  or cares whether the L2 orbit was Sun-Earth or Earth-Moon. source_def.py
  and plots.py are likewise unchanged.

  The L2 file must now be the 6-COLUMN Earth-Moon file (l2_sat_em.txt:
  Az, halo_type, phase0, theta0, Omega_node, i_moon), read by
  data_read_em.l2_sat_data_em.

THE ONE FILE THAT WAS EDITED, AND EXACTLY HOW (per your explicit approval):
  vis_sim.py originally had NO argparse and NO __main__ guard -- msname,
  file_path, and clname were bare hardcoded module-level strings (its
  original lines 37-39) that had to be hand-edited before every run, and
  its body executed unconditionally at import time. The ONLY change made:
  those 4 lines were replaced with a minimal argparse block (same style
  empty_ms.py already used), so vis_sim.py can now be run as
      python3 vis_sim.py <msname> <file_path> <usemod> <model_path>
  Nothing else in vis_sim.py changed -- predictSim(), the HDF5-loading
  loop, the UVW row-count safety check (which still hard-stops on
  mismatch), and the flagdata/predict calls are byte-for-byte identical
  to the original. Full diff is in vis_sim.py's own header comment.

PIPELINE STAGES THIS WRAPPER RUNS, IN ORDER:
  1. dr.sat_data() / dr.l2_sat_data() -- called ONLY to read n_earth/n_l2
     counts (n_sat = n_earth + n_l2, confirmed = Earth sats + L2 sats),
     since empty_ms.py needs n_sat as an input BEFORE the MS is built, and
     main_script.process_satellite() does not return these counts to its
     caller (it returns None). This is a read-only duplicate call against
     the SAME unmodified data_read.py functions, so n_sat is guaranteed
     consistent with what process_satellite() uses internally.
  2. main_script.process_satellite(...) -- same call main_script.py's own
     __main__ block makes in 's' mode; runs data_read -> source_def ->
     coordinate_transform -> uvw_eval, writing the pipeline HDF5.
  3. HDF5 filename reconstructed via the EXACT f-string copied verbatim
     from uvw_eval.py (uvw_cal_s): f'uvw_{RA}_{DEC}_6e_L2_604800.h5'.
     Existence is verified on disk before continuing (hard stop if not
     found, never a silent path mismatch).
  4. plots.PLOT() -- imported unmodified -- called via the verified '..'-
     relative path trick so its hardcoded '/home/tiasha/SVLBI_L2_new/'
     prefix resolves through to wherever the HDF5 actually was written
     (requires that directory to be a sibling of SVLBI_L2_new, checked and
     enforced; raises rather than silently reading the wrong file).
  5. empty_ms.py run as a SUBPROCESS (not imported -- its argparse runs
     unconditionally at module level with no __main__ guard, so importing
     it would consume this wrapper's own sys.argv). n_sat from step 1,
     endtime/int_time from the single combined input, n_chan computed as
     Bandwidth / Frequency_resolution per your own prompt.pdf annotation.
  6. vis_sim.py (the edited copy, see above) run as a subprocess, given
     the MS path just created, the HDF5 path from step 3, and the
     usemod/model_path (.cl or .im) you specify.

WHAT YOU STILL DO BY HAND: launch `casa`, then type your tclean(...) call,
using the exact MS path this wrapper printed at the end.
============================================================================
"""

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt

# Import the existing, UNMODIFIED pipeline modules.
import main_script_em as ms   # EM: uses uvw_eval_em + coordinate_transform_em
import plots as pl            # uses pl.PLOT  (UNCHANGED)
#import data_read as dr        # Earth sats: UNCHANGED reader (n_earth count)
import data_read_em as drem   # EM: L2 sats (6-column file, n_l2 count)
                               # (read-only duplicate of what process_satellite
                               # already reads internally; data_read.py itself
                               # is unmodified)


# Directory that will hold the generated HDF5 (and, by default, the PNGs).
# Sibling of /home/tiasha/SVLBI_L2_new/, per your confirmed choice.
DEFAULT_OUTPUT_DIR = "/home/tiasha/SVLBI_HDF5"

# The hardcoded prefix plots.py's PLOT() concatenates onto whatever filename
# it is given (copied verbatim from plots.py line 56, for path-math purposes
# only -- plots.py itself is never edited).
_PLOTS_PY_HARDCODED_PREFIX = "/home/tiasha/SVLBI_L2_new/"


def list_h5_files(directory):
    """Snapshot of .h5 filenames currently in directory (not full paths)."""
    if not os.path.isdir(directory):
        return set()
    return {f for f in os.listdir(directory) if f.endswith('.h5')}


def discover_new_hdf5(output_dir, files_before):
    """
    Find the HDF5 file uvw_cal_s() just wrote, by set difference rather than
    by guessing its hardcoded filename suffix.

    WHY THIS REPLACED THE EARLIER APPROACH: uvw_eval.py's filename f-string
    (uvw_cal_s, line 219) contains a hardcoded literal suffix -- but that
    literal has been observed to differ between the copy in this project
    ('...6e_L2_604800.h5') and the live file on the actual server
    ('...6e_L2_172800.h5'), confirmed directly via grep on the live machine.
    Since that hardcoded text is evidently hand-edited between runs (outside
    this wrapper's control, and uvw_eval.py is off-limits to modify), trying
    to predict the filename string is fragile and breaks the moment it's
    changed again. Discovering the real file by what NEWLY appeared on disk
    is correct regardless of whatever that literal currently says.
    """
    files_after = list_h5_files(output_dir)
    new_files = files_after - files_before
    if len(new_files) == 0:
        raise RuntimeError(
            f"[wrapper] ERROR: uvw_cal_s() finished but no new .h5 file "
            f"appeared in {output_dir}. Files present before: "
            f"{sorted(files_before) or '(none)'}; after: "
            f"{sorted(files_after) or '(none)'}. The HDF5 was not written "
            "where expected -- check uvw_eval.py's filename logic and "
            "confirm the wrapper's chdir into output_dir actually took effect."
        )
    if len(new_files) > 1:
        raise RuntimeError(
            f"[wrapper] ERROR: more than one new .h5 file appeared in "
            f"{output_dir} during this run: {sorted(new_files)}. Cannot "
            "safely determine which one uvw_cal_s() just wrote. This "
            "should not happen in a normal single run -- please check "
            "manually and clear unexpected files from output_dir."
        )
    return new_files.pop()


def relative_path_for_plots_py(output_dir, hdf5_filename):
    """
    Build a 'file' argument such that, when plots.py internally evaluates
        '/home/tiasha/SVLBI_L2_new/' + file
    the OS resolves it to <output_dir>/<hdf5_filename>.

    Requires output_dir to be a sibling of /home/tiasha/SVLBI_L2_new/ under
    the same parent directory; this script checks that and fails loudly
    (no silent wrong-path reads) if it is not.
    """
    parent_of_hardcoded = os.path.dirname(os.path.normpath(_PLOTS_PY_HARDCODED_PREFIX))
    parent_of_output = os.path.dirname(os.path.normpath(output_dir))

    if parent_of_hardcoded != parent_of_output:
        raise RuntimeError(
            "Cannot safely call the unmodified plots.PLOT(): its hardcoded "
            f"read prefix is '{_PLOTS_PY_HARDCODED_PREFIX}' (parent dir "
            f"'{parent_of_hardcoded}'), but your output_dir's parent is "
            f"'{parent_of_output}'. These must match (sibling directories) "
            "for the '..' relative-path trick to resolve correctly. Either "
            "change --output_dir to a sibling of SVLBI_L2_new under "
            f"'{parent_of_hardcoded}', or this wrapper cannot reach the file "
            "without editing plots.py (which was explicitly ruled out)."
        )

    rel_up = os.path.relpath(output_dir, start=os.path.normpath(_PLOTS_PY_HARDCODED_PREFIX))
    # rel_up will look like '../SVLBI_HDF5'
    return os.path.join(rel_up, hdf5_filename)


def parse_freq_string(s):
    """
    Parse a CASA-style frequency/bandwidth string like '50MHz', '10kHz',
    '604800Hz' into a float in Hz. This logic does NOT exist anywhere in
    empty_ms.py (confirmed: it passes Bandwidth/Frequency_resolution
    straight through to CASA's sm.setspwindow() as raw strings, letting
    CASA parse the unit suffix itself -- it never computes N_chan from
    them). This parser exists ONLY so this wrapper can compute
    N_chan = Bandwidth / Frequency_resolution, per your own prompt.pdf
    annotation that N_chan should not be re-asked of the user. It only
    recognizes the unit suffixes that actually appear in empty_ms.py's own
    help text and your prompt.pdf example (Hz, kHz, MHz, GHz) -- any other
    suffix raises an explicit error rather than guessing.
    """
    s = s.strip()
    units = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1.0}
    for suffix, mult in units.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    raise ValueError(
        f"Cannot parse frequency string '{s}': expected a numeric value "
        f"followed by one of {list(units.keys())} (matching empty_ms.py's "
        "own documented examples, e.g. '50MHz', '10kHz')."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combined single-input wrapper: data_read -> source_def -> "
            "coordinate_transform -> uvw_eval -> main_script -> plots, "
            "in one terminal call. CASA stages (empty_ms.py, vis_sim.py, "
            "tclean) are NOT included here, by design."
        )
    )
    # Same positional order as main_script.py's own argparse (time, int_time,
    # RA, DEC, then the 's' subcommand's satellite_file, l2_file), per your
    # confirmed preference. 'freq' is the one genuinely new input, since it
    # exists only in plots.py's CLI today.
    parser.add_argument('time', type=int, help='Observation time (s).')
    parser.add_argument('int_time', type=int, help='Integration time (s).')
    parser.add_argument('RA', type=str, help='Right Ascension, h:m:s format.')
    parser.add_argument('DEC', type=str, help='Declination, d:m:s format.')
    parser.add_argument('satellite_file', type=str,
                         help='Earth-bound satellite file (e.g. sat_6e.txt).')
    parser.add_argument('l2_file', nargs='?', default=None, type=str,
                         help='Optional Earth-Moon L2 file, 6-column (e.g. l2_sat_em.txt).')   # EM
    parser.add_argument('freq', type=float,
                         help='Central frequency in Hz, e.g. 55e6 (needed by plots.PLOT).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                         help=f'Where the HDF5 and PNGs are saved (default: {DEFAULT_OUTPUT_DIR}).')

    # ---- empty_ms.py inputs (per prompt.pdf: n_sat, endtime, int_time are
    #      explicitly NOT re-asked of the user -- they're derived/reused
    #      from the inputs already given above) ----
    parser.add_argument('msfile', type=str,
                         help='Path/name for the empty MS file (e.g. /home/tiasha/SVLBI_L2_new/EM_L2_6i.ms). '
                              'You type this explicitly, per your confirmed preference.')
    parser.add_argument('Band_name', type=str, help="Band name, e.g. 'band0'.")
    parser.add_argument('Direction', type=str,
                         help='Phasecenter, e.g. "J2000 00h00m00 00d00m00".')
    parser.add_argument('Frequency', type=str, help="Starting frequency, e.g. '50MHz'.")
    parser.add_argument('Frequency_resolution', type=str, help="Freq resolution, e.g. '10kHz'.")
    parser.add_argument('Bandwidth', type=str, help="Bandwidth, e.g. '10MHz'.")
    parser.add_argument('Stokes_Parameters', type=str, help="Stokes, e.g. 'RR'.")

    # ---- vis_sim.py inputs (per your confirmed approval to add argparse) ----
    parser.add_argument('usemod', type=str, choices=['cl', 'im'],
                         help="Sky model type for vis_sim.py: 'cl' (component list) or 'im' (image).")
    parser.add_argument('model_path', type=str,
                         help='Path to the .cl or .im sky-model file for vis_sim.py.')

    # ---- EM dynamics model: CR3BP (default) or ER3BP (elliptic, periodic halo) ----
    parser.add_argument('--model', type=str, default='cr3bp', choices=['cr3bp','er3bp'],
                         help="L2 dynamics: 'cr3bp' (circular, default) or 'er3bp' "
                              "(elliptic; periodic double-revolution halo per "
                              "Ferrari & Lavagna 2021). In er3bp mode the halo "
                              "amplitude is fixed by the 2:1 resonance and l2_sat_em "
                              "Az is ignored.")
    parser.add_argument('--er3bp_e', type=float, default=0.0554,
                         help='Lunar eccentricity for ER3BP mode (default 0.0554, '
                              'Ferrari & Lavagna Earth-Moon value).')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- n_sat: Earth-bound sats + L2 sats, read-only (data_read.py unmodified) ----
    # Confirmed: process_satellite() returns None, so this count is not
    # available from it. We call the SAME dr.sat_data/dr.l2_sat_data
    # functions separately, read-only, purely to get the counts.
    A_tmp, _, _, _, _, _, _, _ = drem.sat_data(args.satellite_file)
    n_earth = len(A_tmp)
    n_l2 = 0
    if args.l2_file:
        Az_tmp, _, _, _, _, _ = drem.l2_sat_data_em(args.l2_file)   # EM: 6 returns
        n_l2 = len(Az_tmp)
    n_sat = n_earth + n_l2
    print(f"[wrapper] n_sat = {n_earth} Earth-bound + {n_l2} L2 = {n_sat}")
    print(f"[wrapper] L2 dynamics model = {args.model.upper()}"
          + (f" (e={args.er3bp_e})" if args.model=='er3bp' else ""))

    # ---- Stage 1: data_read -> source_def -> coordinate_transform -> uvw_eval ----
    # uvw_eval.py's h5py.File(filename, 'a') (line 224) uses a bare relative
    # filename, so it writes into whatever the CWD is at call time. We chdir
    # into output_dir so the HDF5 lands exactly there, with NO change to
    # uvw_eval.py itself.
    original_cwd = os.getcwd()
    h5_files_before = list_h5_files(args.output_dir)  # snapshot BEFORE uvw_cal_s runs
    os.chdir(args.output_dir)
    try:
        print(f"[wrapper] Running main_script.process_satellite(...) "
              f"(writes HDF5 into {args.output_dir})")
        ms.process_satellite(
            args.satellite_file, args.time, args.int_time,
            args.RA, args.DEC, file_l2=args.l2_file,
            model=args.model, er3bp_e=args.er3bp_e          # EM/ER3BP
        )
    finally:
        os.chdir(original_cwd)

    # Discover the real filename by what NEWLY appeared, rather than
    # guessing uvw_eval.py's hardcoded suffix (see discover_new_hdf5 for why).
    try:
        hdf5_filename = discover_new_hdf5(args.output_dir, h5_files_before)
    except RuntimeError as e:
        sys.exit(str(e))
    hdf5_full_path = os.path.join(args.output_dir, hdf5_filename)
    print(f"[wrapper] HDF5 discovered at: {hdf5_full_path}")

    # ---- Stage 2: plots.py's PLOT(), called unmodified ----
    file_arg_for_plot = relative_path_for_plots_py(args.output_dir, hdf5_filename)
    print(f"[wrapper] Calling plots.PLOT() with file='{file_arg_for_plot}' "
          f"(resolves through plots.py's hardcoded prefix to {hdf5_full_path})")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax, handles, labels = pl.PLOT(file_arg_for_plot, args.freq, args.RA, args.DEC, ax=ax)

    # Replicate plots.py's own __main__ save block exactly, but write into
    # output_dir rather than the bare CWD-relative names plots.py uses.
    plt.tight_layout()
    plot_png_path = os.path.join(args.output_dir, "uv_plot.png")
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    plt.close()

    fig_leg = plt.figure(figsize=(6, 4))
    fig_leg.legend(
        handles, labels, loc='center', fontsize=22,
        frameon=True, ncol=1, markerscale=6
    )
    plt.axis('off')
    legend_png_path = os.path.join(args.output_dir, "uv_legend.png")
    plt.savefig(legend_png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[wrapper] Saved: {plot_png_path}")
    print(f"[wrapper] Saved: {legend_png_path}")

    # ---- Stage 3: empty_ms.py, run as a SUBPROCESS ----
    # empty_ms.py HAS its own argparse (confirmed: lines 96-111) but NO
    # __main__ guard -- parser.parse_args() runs at module import time. That
    # means it CANNOT be `import`-ed here (it would try to consume this
    # wrapper's own sys.argv). It CAN be invoked exactly as you already do
    # by hand: `python3 empty_ms.py <args>`. So we shell out to it, byte-for
    # -byte unmodified, via subprocess -- this is not a workaround, it is
    # simply automating the same terminal command you already run.
    n_chan = int(round(parse_freq_string(args.Bandwidth) / parse_freq_string(args.Frequency_resolution)))
    print(f"[wrapper] N_chan = Bandwidth/Frequency_resolution = "
          f"{parse_freq_string(args.Bandwidth)}/{parse_freq_string(args.Frequency_resolution)} = {n_chan}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    empty_ms_path = os.path.join(script_dir, "empty_ms.py")

    empty_ms_cmd = [
        sys.executable, empty_ms_path,
        args.msfile, str(n_sat), str(args.time), str(args.int_time),
        args.Band_name, args.Direction, args.Frequency,
        args.Frequency_resolution, args.Bandwidth, str(n_chan),
        args.Stokes_Parameters,
    ]
    print(f"[wrapper] Running empty_ms.py: {' '.join(empty_ms_cmd)}")
    result = subprocess.run(empty_ms_cmd)
    if result.returncode != 0:
        sys.exit(f"[wrapper] ERROR: empty_ms.py exited with code {result.returncode}. "
                  "Stopping before vis_sim.py -- an empty/incomplete MS must not be "
                  "fed forward.")
    print(f"[wrapper] empty_ms.py finished. MS at: {args.msfile}")

    # ---- Stage 4: vis_sim.py (the version with argparse added, see file
    #      header for the exact, minimal diff against the original), also
    #      run as a SUBPROCESS for the same reason as empty_ms.py: its body
    #      executes unconditionally at import time. ----
    vis_sim_path = os.path.join(script_dir, "vis_sim.py")
    vis_sim_cmd = [
        sys.executable, vis_sim_path,
        args.msfile, hdf5_full_path, args.usemod, args.model_path,
    ]
    print(f"[wrapper] Running vis_sim.py: {' '.join(vis_sim_cmd)}")
    result = subprocess.run(vis_sim_cmd)
    if result.returncode != 0:
        sys.exit(f"[wrapper] ERROR: vis_sim.py exited with code {result.returncode}.")

    print("\n[wrapper] Pipeline complete through vis_sim.py.")
    print(f"[wrapper] MS ready for tclean: {args.msfile}")
    print("[wrapper] Remaining step (by design, run by hand, per your instruction):")
    print("    1. Launch CASA:  casa")
    print("    2. At the CASA<1> prompt, run your tclean(...) call with")
    print(f"       vis='{args.msfile}'")


if __name__ == "__main__":
    main()