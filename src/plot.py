""" 
=================================
Description
=================================
This file explicitly focuses on plotting functions
This includes:
- `calibrations()`: plotting calibrations within the same timeline
"""

# =================================
# Imports
# =================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================
# Plot calibrations
# =================================

def calibrations(
    tdf, 
    cdf, 
    tdf_colname:str='unix_ms', 
    cdf_colname:str='unix_ms',
    outpath:str=None,
    show:bool=True
):
    """ 
    --- Collect first timestamps from calibration files ---
    """

    # Extract timestamps
    x = tdf[tdf_colname].to_numpy()
    _y = cdf[[cdf_colname, 'event', 'trial_id']]
    y = _y[_y['event']=='Start']

    # Start, end, and trial start times
    start_time = x.min()
    end_time = x.max()
    markers = [(f"Trial {i}:\n{t}", t) for t,i in zip(y[cdf_colname],y['trial_id'])]

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(x, [0]*len(x), alpha=0.2, label="VR time")

    # Add vertical markers for each smaller file
    for name, t in markers:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        plt.text(t, 0.1, name, rotation=90, verticalalignment='bottom', fontsize=8)

    # Add a vertical marker for the start and ends too.
    plt.axvline(x=start_time, color='blue', linestyle='--', alpha=0.7)
    plt.text(start_time, 0.1, f"Eye Start\n{start_time}", rotation=90, verticalalignment='bottom', fontsize=7.5)

    # Render other stuff
    plt.xlim(start_time, end_time)
    plt.xlabel(tdf_colname)
    plt.yticks([])
    plt.title(f"Timeline of Eye Data to Calibrations:")
    plt.legend()
    plt.tight_layout()
    if outpath is not None:        
        outdir, outname = os.path.split(outpath)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outpath, bbox_inches="tight", dpi=300)
    if show:    plt.show()
    else:       plt.close()


