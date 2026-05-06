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
            'height':150,
        },
        {   'title':"Force", 
            'axis_title':'m/s/s',
            'features':["force", "force_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'range':[-5.0, 5.0],
            'height':150,
        },
        {   'title':"Intended Heading", 
            'axis_title':'radians',
            'features':["move_heading_intent","move_heading_intent_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'], 
            'range':[-4.25, 4.25],
            'height':150,
        },
        {   'title':"Int. Head. to Goal", 
            'axis_title':'Dot Prod. (0:1)',
            'features':["move_heading_intent_rel_goal", "move_heading_intent_rel_goal_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-2, 2],
            'height':150,
        },
        {   'title':"Distance to Confederate", 
            'axis_title':'m',
            'features':["distance_to_confederate", "distance_to_confederate_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[0, 15],
            'height':150,
        },
        {   'title':"Ahead", 
            'axis_title':'Dot Prod. (-1:1)',
            'features':["ahead", "ahead_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-1.05, 1.05],
            'height':150,
        },
        {   'title':"Side", 
            'axis_title':'Dot Prod. (-1:1)',
            'features':["side", "side_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-1.05, 1.05],
            'height':150,
        },
    ],
    trajectory_height:int = 150,
    spacing:float = 0.025,
    fig_title:str = None,
    show:bool = True,
    outpath:str = None,
):
    # Calculate figure height
    row_pixel_heights = [f["height"] for f in features] + [trajectory_height]
    n_rows = len(row_pixel_heights)
    content_height = sum(row_pixel_heights)
    total_height = content_height / (1 - spacing * (n_rows - 1))
    row_heights = [h / content_height for h in row_pixel_heights]
    
    # Build figure
    df = move_df.sort_values("unix_ms")
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=spacing,
        row_heights=row_heights,
        subplot_titles= [f['title'] for f in features] + ["trajectory"]
    )
    fig.update_layout(height=total_height)

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
    if fig_title is None:
        fig_title = f"Trial {df['trial_id'].iloc[0]}"
    fig.update_layout(
        height=total_height,
        title=fig_title,
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
                    height=total_height,
                    scale=2
                )
                #print(f"Saved static image → {outpath}")
            except Exception as e:
                print("⚠️ Could not save static image.")
                print("Install kaleido: pip install kaleido")
                print(f"Error: {e}")


# =================================
# Plot Trajectory Playback
# =================================

def trajectories_playback(
    move_df: pd.DataFrame,
    features = [
        {   'title':"Speed", 
            'axis_title':'m/s',
            'features':["speed","speed_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'range':[0.0, 1.5],
            'height':150,
        },
        {   'title':"Force", 
            'axis_title':'m/s/s',
            'features':["force", "force_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'range':[-5.0, 5.0],
            'height':150,
        },
        {   'title':"Intended Heading", 
            'axis_title':'radians',
            'features':["move_heading_intent","move_heading_intent_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'], 
            'range':[-4.25, 4.25],
            'height':150,
        },
        {   'title':"Int. Head. to Goal", 
            'axis_title':'Dot Prod. (0:1)',
            'features':["move_heading_intent_rel_goal", "move_heading_intent_rel_goal_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-2, 2],
            'height':150,
        },
        {   'title':"Distance to Confederate", 
            'axis_title':'m',
            'features':["distance_to_confederate", "distance_to_confederate_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[0, 15],
            'height':150,
        },
        {   'title':"Ahead", 
            'axis_title':'Dot Prod. (-1:1)',
            'features':["ahead", "ahead_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-1.05, 1.05],
            'height':150,
        },
        {   'title':"Side", 
            'axis_title':'Dot Prod. (-1:1)',
            'features':["side", "side_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'range':[-1.05, 1.05],
            'height':150,
        },
    ],
    trajectory_height: int = 150,
    spacing: float = 0.025,
    fig_title: str = None,
):
    df = move_df.sort_values("unix_ms").reset_index(drop=True)

    # --- Layout sizing ---
    row_pixel_heights = [f["height"] for f in features] + [trajectory_height]
    n_rows = len(row_pixel_heights)
    content_height = sum(row_pixel_heights)
    total_height = content_height / (1 - spacing * (n_rows - 1))
    row_heights = [h / content_height for h in row_pixel_heights]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=spacing,
        row_heights=row_heights,
        subplot_titles=[f['title'] for f in features] + ["trajectory"]
    )

    # --- INITIAL TRACES (EMPTY) ---
    trace_map = []  # keeps track of trace ordering

    # Time series traces
    for i, feature in enumerate(features, start=1):
        for f, o, c in zip(feature['features'], feature['opacity'], feature['color']):
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    opacity=o,
                    line=dict(color=c),
                    showlegend=False,
                ),
                row=i,
                col=1
            )
            trace_map.append(("timeseries", f))

    # Trajectory trace
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(
                color=[],
                colorscale="Jet",
                size=4,
                cmin=0,
                cmax=df["speed"].max() if "speed" in df else None
            ),
            showlegend=False,
        ),
        row=n_rows,
        col=1
    )
    trace_map.append(("trajectory", None))

    # --- AXIS FORMATTING ---
    for i, feature in enumerate(features, start=1):
        if 'range' in feature:
            fig.update_yaxes(range=feature['range'], row=i, col=1)
        if 'axis_title' in feature:
            fig.update_yaxes(title_text=feature['axis_title'], row=i, col=1)

    # Trajectory axes
    fig.update_xaxes(title_text="X", row=n_rows, col=1, range=[-5, 5])
    fig.update_yaxes(title_text="Y", row=n_rows, col=1, range=[-4, 0])

    # --- FRAMES ---
    frames = []

    for i in range(len(df)):
        frame_data = []

        # Build traces in EXACT same order
        for kind, col_name in trace_map:
            if kind == "timeseries":
                frame_data.append(
                    go.Scatter(
                        x=df["unix_ms"][:i+1],
                        y=df[col_name][:i+1],
                    )
                )
            elif kind == "trajectory":
                frame_data.append(
                    go.Scatter(
                        x=df["x_pos"][:i+1],
                        y=df["y_pos"][:i+1],
                        marker=dict(
                            color=df["speed"][:i+1] if "speed" in df else None,
                            colorscale="Jet",
                            size=4,
                        ),
                    )
                )

        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames

    # --- SLIDER ---
    sliders = [{
        "steps": [
            {
                "args": [
                    [str(i)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate"
                    }
                ],
                "label": str(i),
                "method": "animate"
            }
            for i in range(len(frames))
        ],
        "x": 0.1,
        "y": 0,
        "len": 0.9
    }]

    # --- PLAY / PAUSE ---
    updatemenus = [{
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0},
                        "mode": "immediate"
                    }
                ]
            }
        ]
    }]

    # --- TITLE ---
    if fig_title is None:
        fig_title = f"Trial {df['trial_id'].iloc[0]}"

    fig.update_layout(
        height=total_height,
        title=fig_title,
        sliders=sliders,
        updatemenus=updatemenus,
        showlegend=False
    )

    fig.show(renderer="browser")