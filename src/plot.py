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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# =================================
# Plot Trajectories
# =================================

def trajectories(
    move_df:pd.DataFrame,
    features = [
        {   'title':"Speed", 
            'axis_title':'m/s',
            'features':["speed","speed_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'range':[0.0, 1.5],
        },
        {   'title':"Force", 
            'axis_title':'m/s/s',
            'features':["force", "force_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'range':[-5.0, 5.0],
        },
        {   'title':"Intended Heading", 
            'axis_title':'radians',
            'features':["move_heading_intent","move_heading_intent_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'], 
            'range':[-4.25, 4.25],
        },
        {   'title':"Int. Head. to Goal", 
            'axis_title':'Dot Prod. (0:1)',
            'features':["move_heading_intent_toward_goal", "move_heading_intent_toward_goal_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-0.05, 1.05],
        },
    ],
    show:bool = True,
    outpath:str = None,
):
    # Build figure
    df = move_df.sort_values("unix_ms")
    n_rows = len(features) + 1  # +1 for trajectory
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.075,
        subplot_titles= [f['title'] for f in features] + ["trajectory"]
    )

    # --- Time series ---
    t = df["unix_ms"]
    for i, feature in enumerate(features, start=1):
        title = feature['title']
        for f,o,c in zip(feature['features'],feature['opacity'],feature['color']):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df[f],
                    mode="lines",
                    opacity=o,
                    line=dict(color=c),
                    name=title,
                ),
                row=i,
                col=1
            )
        if 'range' in feature:      fig.update_yaxes(row=i, col=1, range=feature['range'])
        if 'axis_title' in feature: fig.update_yaxes(row=i, col=1, title_text=feature['axis_title'])

    # --- Trajectory ---
    traj_row = n_rows
    fig.add_trace(
        go.Scatter(
            x=df["x_pos"],
            y=df["y_pos"],
            name="trajectory",
            mode="markers",
            marker=dict(
                color=df["speed"],
                colorscale=[
                    [0.0, "rgb(0,0,255)"],      # blue (slow)
                    [0.5, "rgb(255,255,0)"],    # yellow (medium)
                    [1.0, "rgb(255,0,0)"]       # red (fast)
                ],
                cmin=0,
                cmax=df["speed"].max(),
                size=3,
                colorbar=dict(title="Speed")
            ),
        ),
        row=traj_row,
        col=1
    )

    fig.update_xaxes(title_text="X", row=traj_row, col=1, range=[-5.0, 5.0])
    #fig.update_yaxes(title_text="Y", row=traj_row, col=1, range=[-5.0, 2.5])
    fig.update_yaxes(title_text="Y", row=traj_row, col=1, range=[-4, 0])

    # Optional: keep spatial proportions correct
    #fig.update_yaxes(scaleanchor="x", row=traj_row, col=1)

    # --- Layout ---
    fig.update_layout(
        height=900,
        title=f"Trial {df['trial_id'].iloc[0]}",
        showlegend=False
    )

    # Showing
    if show:
        fig.show(renderer="browser")

    # Saving
    if outpath is not None:
        # Initialize directory
        dirpath = os.path.dirname(outpath)
        os.makedirs(dirpath, exist_ok=True)

        # Define extension
        ext = os.path.splitext(outpath)[1].lower()

        # writing by extension
        if ext == ".html":
            fig.write_html(outpath)
            #print(f"Saved interactive dashboard → {outpath}")
        else:
            try:
                fig.write_image(
                    outpath,
                    width=900,
                    height=900,
                    scale=2
                )
                #print(f"Saved static image → {outpath}")
            except Exception as e:
                print("⚠️ Could not save static image.")
                print("Install kaleido: pip install kaleido")
                print(f"Error: {e}")
