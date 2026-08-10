import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import source_def as sc
 
# ---------------- MAIN PLOT FUNCTION ----------------
def PLOT(file, freq, RA, DEC, ax=None):
 
    # Define colors
    #colour = ['b', 'g', 'm']   ##### for 3 satellites
    #colour = ['b', 'g', 'r', 'c', 'm', 'y', 'olive', 'pink', 'brown', 'orange'] ##### for 5 satellites 
    #colour = ['b', 'g', 'r','m', 'y','pink'] ### for 4 satellites 
    # If no axes provided, create a new figure and axes (for standalone usage)
 
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
 
    # ---------------- STYLE ----------------
    plt.rcParams.update({
        'axes.linewidth': 4.5,          # ⬅️ thicker axis
        'xtick.major.width': 3.5,
        'ytick.major.width': 3.5,
        'xtick.minor.width': 2.5,
        'ytick.minor.width': 2.5,
        'xtick.major.size': 12,
        'ytick.major.size': 12,
        'xtick.labelsize': 30,          # ⬅️ bigger numbers
        'ytick.labelsize': 30,
        'font.size': 30,
        'axes.labelweight': 'bold'
    })
 
    # ❌ REMOVE AXIS LABELS
    ax.set_xlabel('')
    ax.set_ylabel('')
 
    # ---------------- AXIS THICKNESS ----------------
    for spine in ax.spines.values():
        spine.set_linewidth(4)          # ⬅️ thicker spines
 
    # ---------------- TICKS ----------------
    ax.tick_params(axis='both', which='major',
                   labelsize=30, width=3.5, length=10, direction='in')
    ax.tick_params(axis='both', which='minor',
                   width=2.5, length=5, direction='in')
 
    ax.minorticks_on()
    ax.grid(True, linewidth=1, alpha=0.6)
 
    # ---------------- DATA ----------------
    c = 3e8
    lamb = c / freq
 
    f1 = h5py.File('/home/tiasha/SVLBI_L2_new/' + file, 'r')
    groups = list(f1.keys())

    zorders = list(range(1, len(groups) + 1))  
  
    import matplotlib.cm as cm
    # First 10 baselines keep their original fixed colors (as before).
    base_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'olive', 'pink', 'brown', 'orange']
    # Any baseline beyond the first 10 gets a distinct auto-generated color.
    n_extra = max(len(groups) - len(base_colours), 0)
    extra_colours = [cm.tab20(k / max(n_extra - 1, 1)) for k in range(n_extra)]
    colour = base_colours + extra_colours    

    for idx, group_name in enumerate(groups):
        data_f = []
        k = 0
 
        for dataset_name, dataset in f1[group_name].items():
            if k < 2:
                data_in = np.squeeze(dataset[()])
                data_f.append(data_in)
                k += 1
            else:
                break
 
        data_f1 = np.asarray(data_f)
        x_vals = data_f1[0, :] / lamb
        y_vals = data_f1[1, :] / lamb
 
        sc1 = ax.scatter(
            x_vals, y_vals,
            color=colour[idx % len(colour)],
            s=1.0,
            alpha=0.9,
            zorder=zorders[idx],
            label=f"{group_name}"
        )
 
        ax.scatter(
            -x_vals, -y_vals,
            color=colour[idx % len(colour)],
            s=1.0,
            alpha=0.9,
            zorder=zorders[idx]
        )
 
    f1.close()
 
    handles, labels = ax.get_legend_handles_labels()
    return ax, handles, labels
 
 
# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description="Plot the u-v points")
    sub = parser.add_subparsers(dest='mode')
 
    parser.add_argument('f', type=float)
    parser.add_argument('ra', type=str)
    parser.add_argument('dec', type=str)
 
    parser_s = sub.add_parser('s')
    parser_s.add_argument('uvw_satellite_file', type=str)
 
    parser_g = sub.add_parser('g')
    parser_g.add_argument('uvw_antenna_file', type=str)
 
    parser_combo = sub.add_parser('s+g')    
    parser_combo.add_argument('uvw_satellite_file', type=str)
    parser_combo.add_argument('uvw_antenna_file', type=str)
    parser_combo.add_argument('uvw_space_antenna_file', type=str)
 
    args = parser.parse_args()
    freq = args.f
 
    fig, ax = plt.subplots(figsize=(10, 10))
 
    if args.mode == 's':
        ax, handles, labels = PLOT(args.uvw_satellite_file, freq, args.ra, args.dec, ax=ax)
 
    elif args.mode == 'g':
        ax, handles, labels = PLOT(args.uvw_antenna_file, freq, args.ra, args.dec, ax=ax)
 
    elif args.mode == 's+g':
        ax, handles, labels = PLOT(args.uvw_satellite_file, freq, args.ra, args.dec, ax=ax)
        ax, handles, labels = PLOT(args.uvw_antenna_file, freq, args.ra, args.dec, ax=ax)
        ax, handles, labels = PLOT(args.uvw_space_antenna_file, freq, args.ra, args.dec, ax=ax)
 
    plt.tight_layout()
    plt.savefig("uv_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
 
    # ---------------- LEGEND ----------------
    fig_leg = plt.figure(figsize=(6, 4))
    fig_leg.legend(
        handles,
        labels,
        loc='center',
        fontsize=22,
        frameon=True,
        ncol=1,
        markerscale=6
    )
 
    plt.axis('off')
    plt.savefig("uv_legend.png", dpi=300, bbox_inches='tight')
    plt.show()