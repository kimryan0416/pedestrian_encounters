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
            'xaxis_title':'Time (ms)',
            'yaxis_title':'m/s',
            'x':['unix_ms','unix_ms'],
            'y':["speed","speed_lowpass"],
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'yrange':[0.0, 1.5],
            'width': 500,
            'height':150,
            'row':1, 'col':1,
        },
        {   'title':"Force", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'m/s/s',
            'x':['unix_ms','unix_ms'],
            'y':["force", "force_lowpass"],
            'opacity':[0.25, 1.0],
            'color':['red','blue'],
            'yrange':[-5.0, 5.0],
            'width': 500,
            'height':150,
            'row':1, 'col':2,
        },
        {   'title':"Intended Heading", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'radians',
            'x':['unix_ms','unix_ms'],
            'y':["move_heading_intent","move_heading_intent_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'], 
            'yrange':[-4.25, 4.25],
            'width': 500,
            'height':150,
            'row':2, 'col':1,
        },
        {   'title':"Int. Head. to Goal", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'Dot Prod. (0:1)',
            'x':['unix_ms','unix_ms'],
            'y':["move_heading_intent_rel_goal", "move_heading_intent_rel_goal_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'yrange':[-2, 2],
            'width': 500,
            'height':150,
            'row':2, 'col':2,
        },
        {   'title':"Distance to Confederate", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'m',
            'x':['unix_ms','unix_ms'],
            'y':["distance_to_confederate", "distance_to_confederate_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'yrange':[0, 20],
            'width': 500,
            'height':150,
            'row':3, 'col':1,
        },
        {   'title':"Ahead", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'Dot Prod. (-1:1)',
            'x':['unix_ms','unix_ms'],
            'y':["ahead", "ahead_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'yrange':[-1.05, 1.05],
            'width': 500,
            'height':150,
            'row':4, 'col':1,
        },
        {   'title':"Side", 
            'xaxis_title':'Time (ms)',
            'yaxis_title':'Dot Prod. (-1:1)',
            'x':['unix_ms','unix_ms'],
            'y':["side", "side_lowpass"], 
            'opacity':[0.25, 1.0],
            'color':['red','blue'],    
            'yrange':[-1.05, 1.05],
            'width': 500,
            'height':150,
            'row':4, 'col':2,
        },
        {   'title':"Trajectory", 
            'xaxis_title':'X',
            'yaxis_title':'Y',
            'x':['x_pos_lowpass', 'c_x_pos_lowpass'],
            'y':['y_pos_lowpass', 'c_y_pos_lowpass'], 
            'opacity':[1.0, 0.5],
            'color':['speed', 'gray'],    
            'xrange':[-5.0, 5.0],
            'yrange':[-4, 0],
            'width': 500,
            'height':200,
            'row':5, 'col':1,
        },
    ],
    spacing:float = 0.075,
    fig_title:str = None,
    show:bool = True,
    outpath:str = None,
):  
    # Maximum number of rows and columns
    n_rows = max([f['row'] for f in features])
    n_cols = max([f['col'] for f in features])
    # Calcualte the row heights for the figure grid
    row_heights_raw = []
    for r in range(1, n_rows + 1):
        row_features = [f for f in features if f['row'] == r]
        max_height = max(f.get('height', 1) for f in row_features)
        row_heights_raw.append(max_height)
    content_height = sum(row_heights_raw)                           # The content row heights (not including spacing)
    total_height = content_height / (1 - spacing * (n_rows - 1))    # content row heights + spacing
    row_heights = [h / content_height for h in row_heights_raw]     # Ratio of measured height to content height
    # Same for column widths
    column_widths_raw = []
    for c in range(1, n_cols + 1):
        col_features = [f for f in features if f['col'] == c]
        max_width = max(f.get('width', 1) for f in col_features)
        column_widths_raw.append(max_width)
    content_width = sum(column_widths_raw)
    total_width = content_width / (1 - spacing * (n_cols - 1))
    column_widths = [w / content_width for w in column_widths_raw]
    # Subplot titles
    titles = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            feature = next(
                ( f for f in features if f['row'] == r and f['col'] == c ),
                None
            )
            titles.append(feature['title'] if feature else "")
    
    # Build figure
    df = move_df.sort_values("unix_ms")
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=spacing,
        row_heights=row_heights,
        column_widths=column_widths,
        subplot_titles=titles,
    )

    for feature in features:
        row = feature['row']
        col = feature['col']
        for x,y,o,c in zip(feature['x'], feature['y'], feature['opacity'], feature['color']):
            # if either x or y are NOT columns in the dataframe, then we skip
            if x not in df.columns or y not in df.columns:
                continue    
            # Unique situation: marker must be defined by whether it's an existing column in our dataframe or not.abs
            if c in df.columns:
                # Case 1: This is a column!
                marker_cfg = dict(
                    color=df[c],
                    colorscale=[
                        [0.0, "rgb(0,0,255)"],
                        [0.5, "rgb(255,255,0)"],
                        [1.0, "rgb(255,0,0)"]
                    ],
                    cmin=df[c].min(),
                    cmax=df[c].max(),
                    size=3,
                )
            else:
                # Case 2: Try to treat as a normal color
                marker_cfg = dict(
                    color=c,
                    size=3
                )
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="markers",
                    opacity=o,
                    marker=marker_cfg,
                ),
                row=row,
                col=col
            )
        # Modify tick sizes
        fig.update_xaxes(row=row, col=col, tickfont=dict(size=8))
        fig.update_yaxes(row=row, col=col, tickfont=dict(size=8))
        if 'xrange' in feature:          fig.update_xaxes(row=row, col=col, range=feature['xrange'])
        if 'yrange' in feature:          fig.update_yaxes(row=row, col=col, range=feature['yrange'])
        if 'xaxis_title' in feature:    fig.update_xaxes(row=row, col=col, title_text=feature['xaxis_title'], title_font=dict(size=10), title_standoff=5)
        if 'yaxis_title' in feature:    fig.update_yaxes(row=row, col=col, title_text=feature['yaxis_title'], title_font=dict(size=10), title_standoff=5)

    # Optional: keep spatial proportions correct
    #fig.update_yaxes(scaleanchor="x", row=traj_row, col=1)

    # Update some latent elements
    if fig_title is None:
        fig_title = f"Trial {df['trial_id'].iloc[0]}"
    fig.update_layout(
        title=fig_title,
        height=total_height, 
        width=total_width,
        showlegend=False,
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
            'range':[0, 20],
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