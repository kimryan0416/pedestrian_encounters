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

trajectory_feature_defaults = [
    {   'title':"Speed", 
        'xaxis_title':'Time (ms)', 
        'yaxis_title':'m/s',
        'x':['unix_ms','unix_ms'],  
        'y':["speed","speed_lowpass"],
        'legend': ['Raw Speed', 'Smoothed Speed'],
        'opacity':[0.25, 1.0],      
        'color':['red','blue'],
        'yrange':[0.0, 1.5],
        'width': 500, 
        'height':125,
        'row':1, 
        'col':1,
    },
    {   'title':"Force", 
        'xaxis_title':'Time (ms)',
        'yaxis_title':'m/s/s',
        'x':['unix_ms','unix_ms'],
        'y':["force", "force_lowpass"],
        'legend': ['Raw Accel', 'Smoothed Accel'],
        'opacity':[0.25, 1.0],
        'color':['red','blue'],
        'yrange':[-5.0, 5.0],
        'width': 500,
        'height':125,
        'row':1, 
        'col':2,
    },
    #{   'title':"Interpolated Heading", 
    #    'xaxis_title':'Time (ms)',
    #    'yaxis_title':'Radians',
    #    'x':['unix_ms','unix_ms'],
    #    'y':["move_heading_interp","move_heading_interp_lowpass"], 
    #    'legend': ['Raw Heading', 'Smoothed Heading'],
    #    'opacity':[0.25, 1],
    #    'color':['red','blue'], 
    #    'yrange':[-3.5, 3.5],
    #    'width': 500,
    #    'height':125,
    #    'row':2, 
    #    'col':1,
    #},
    {   'title':"Int. Head. to Goal Axis", 
        'xaxis_title':'Time (ms)',
        'yaxis_title':'Radians',
        'x':['unix_ms','unix_ms'],
        'y':["move_heading_interp_rel_goal", "move_heading_interp_rel_goal_lowpass"], 
        'legend': ['Raw Heading to Goal', 'Smoothed Heading to Goal'],
        'opacity':[0.25, 1],
        'color':['red','blue'],    
        'yrange':[-3.5, 3.5],
        'width': 500,
        'height':125,
        'row':2, 
        'col':1,
    },
    {   'title':"Distance to Confederate", 
        'xaxis_title':'Time (ms)',
        'yaxis_title':'m',
        'x':['unix_ms','unix_ms'],
        'y':["distance_to_confederate", "distance_to_confederate_lowpass"], 
        'legend': ['Raw Distance', 'Smoothed Distance'],
        'opacity':[0.25, 1.0],
        'color':['red','blue'],    
        'yrange':[0, 20],
        'width': 500,
        'height':125,
        'row':2, 
        'col':2,
    },
    {   'title':"Ahead", 
        'xaxis_title':'Time (ms)',
        'yaxis_title':'Dot Prod. (-1:1)',
        'x':['unix_ms','unix_ms'],
        'y':["ahead", "ahead_lowpass"], 
        'legend': ['Raw Ahead', 'Smoothed Ahead'],
        'opacity':[0.25, 1.0],
        'color':['red','blue'],    
        'yrange':[-1.05, 1.05],
        'width': 500,
        'height':125,
        'row':3, 
        'col':1,
    },
    {   'title':"Side", 
        'xaxis_title':'Time (ms)',
        'yaxis_title':'Dot Prod. (-1:1)',
        'x':['unix_ms','unix_ms'],
        'y':["side", "side_lowpass"], 
        'legend': ['Raw Side', 'Smoothed Side'],
        'opacity':[0.25, 1.0],
        'color':['red','blue'],    
        'yrange':[-1.05, 1.05],
        'width': 500,
        'height':125,
        'row':3, 
        'col':2,
    },
    {   'title':"Trajectory", 
        'xaxis_title':'X',
        'yaxis_title':'Y',
        'x':['x_pos_lowpass', 'c_x_pos_lowpass'],
        'y':['y_pos_lowpass', 'c_y_pos_lowpass'], 
        'legend': ["User's Trajectory", "Confederate's Trajectory"],
        'opacity':[1.0, 0.5],
        'color':['speed', 'gray'],    
        'xrange':[-5.0, 5.0],
        'yrange':[-4, 0],
        'width': 500,
        'height':120,
        'row':4, 
        'col':1,
    },
]

def trajectories(
    move_df:pd.DataFrame,
    features = trajectory_feature_defaults,
    spacing:float = 0.1,
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
    features = trajectory_feature_defaults,
    spacing: float = 0.1,
    playback_ms: int = 50,
    fig_title: str = None,
):

    df = move_df.sort_values("unix_ms").reset_index(drop=True)

    # =========================================================
    # GRID SIZE
    # =========================================================

    n_rows = max(f['row'] for f in features)
    n_cols = max(f['col'] for f in features)

    # =========================================================
    # ROW HEIGHTS
    # =========================================================

    row_heights_raw = []
    for r in range(1, n_rows + 1):
        row_features = [f for f in features if f['row'] == r]
        max_height = max(
            f.get('height', 1)
            for f in row_features
        )
        row_heights_raw.append(max_height)
    content_height = sum(row_heights_raw)
    total_height = content_height / (1 - spacing * (n_rows - 1))
    row_heights = [h / content_height for h in row_heights_raw]

    # =========================================================
    # COLUMN WIDTHS
    # =========================================================

    column_widths_raw = []
    for c in range(1, n_cols + 1):
        col_features = [f for f in features if f['col'] == c ]
        max_width = max( f.get('width', 1) for f in col_features )
        column_widths_raw.append(max_width)
    content_width = sum(column_widths_raw)
    total_width = content_width / (1 - spacing * (n_cols - 1))
    column_widths = [w / content_width for w in column_widths_raw]

    # =========================================================
    # SUBPLOT TITLES
    # =========================================================

    titles = []
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            feature = next(
                (
                    f for f in features
                    if f['row'] == r and f['col'] == c
                ),
                None
            )
            titles.append(feature['title'] if feature else "")

    # =========================================================
    # BUILD FIGURE
    # =========================================================

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=spacing,
        horizontal_spacing=spacing,
        row_heights=row_heights,
        column_widths=column_widths,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    # =========================================================
    # TRACE VISIBILITY MAP
    # =========================================================

    feature_trace_indices = {}
    trace_idx = 0

    # =========================================================
    # ADD STATIC TRACES
    # =========================================================

    cursor_trace_indices = []

    for feature in features:
        row = feature['row']
        col = feature['col']
        title = feature['title']
        feature_trace_indices[title] = []

        # -----------------------------------------------------
        # MAIN STATIC TRACES
        # -----------------------------------------------------

        for x, y, o, c, l in zip(
            feature['x'],
            feature['y'],
            feature['opacity'],
            feature['color'],
            feature['legend'],
        ):
            if x not in df.columns or y not in df.columns:
                continue

            # ---------------------------------------------
            # MARKER / LINE CONFIG
            # ---------------------------------------------

            if c in df.columns:
                marker_cfg = dict(
                    color=df[c],
                    colorscale="Jet",
                    cmin=df[c].min(),
                    cmax=df[c].max(),
                    size=4,
                )
            else:
                marker_cfg = dict(
                    color=c,
                    size=4,
                )
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="markers",
                    opacity=o,
                    marker=marker_cfg,
                    name=l,
                    legendgroup=title,
                    showlegend=True,
                ),
                row=row,
                col=col
            )
            feature_trace_indices[title].append(trace_idx)
            trace_idx += 1

        # -----------------------------------------------------
        # CURSOR LINE
        # -----------------------------------------------------

        if "unix_ms" in df.columns:
            x0 = df["unix_ms"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0],
                    y=feature.get('yrange', [0, 1]),
                    mode="lines",
                    line=dict(
                        color="black",
                        dash="dash",
                        width=1,
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col
            )
            cursor_trace_indices.append(trace_idx)
            trace_idx += 1

        # -----------------------------------------------------
        # AXIS FORMATTING
        # -----------------------------------------------------

        fig.update_xaxes(
            row=row,
            col=col,
            tickfont=dict(size=8)
        )

        fig.update_yaxes(
            row=row,
            col=col,
            tickfont=dict(size=8)
        )

        if 'xrange' in feature:         fig.update_xaxes(row=row, col=col, range=feature['xrange'])
        if 'yrange' in feature:         fig.update_yaxes(row=row, col=col, range=feature['yrange'])
        if 'xaxis_title' in feature:    fig.update_xaxes(row=row, col=col, title_text=feature['xaxis_title'], title_font=dict(size=10), title_standoff=5)
        if 'yaxis_title' in feature:    fig.update_yaxes(row=row, col=col, title_text=feature['yaxis_title'], title_font=dict(size=10), title_standoff=5)

    # =========================================================
    # ANIMATED CURRENT POSITION MARKER
    # =========================================================

    traj_feature = next(
        (
            f for f in features
            if f['title'] == "Trajectory"
        ),
        None
    )


    current_marker_indices = []
    if traj_feature is not None:
        # =====================================================
        # PRIMARY AGENT MARKER
        # =====================================================
        fig.add_trace(
            go.Scatter(
                x=[df["x_pos_lowpass"].iloc[0]],
                y=[df["y_pos_lowpass"].iloc[0]],
                mode="markers",
                marker=dict(
                    color="black",
                    size=10,
                    symbol="circle"
                ),
                name="Agent",
                showlegend=False,
            ),
            row=traj_feature['row'],
            col=traj_feature['col']
        )
        current_marker_indices.append(trace_idx)
        trace_idx += 1
        # =====================================================
        # CONFEDERATE MARKER
        # =====================================================
        fig.add_trace(
            go.Scatter(
                x=[df["c_x_pos_lowpass"].iloc[0]],
                y=[df["c_y_pos_lowpass"].iloc[0]],
                mode="markers",
                marker=dict(
                    color="magenta",
                    size=10,
                    symbol="diamond"
                ),
                name="Confederate",
                showlegend=False,
            ),
            row=traj_feature['row'],
            col=traj_feature['col']
        )
        current_marker_indices.append(trace_idx)
        trace_idx += 1

    # =========================================================
    # FRAMES
    # =========================================================

    frames = []
    for i in range(len(df)):
        frame_updates = []
        # ---------------------------------------------
        # CURSOR LINES
        # ---------------------------------------------
        for feature in features:
            if "unix_ms" not in df.columns:     continue
            x0 = df["unix_ms"].iloc[i]
            frame_updates.append(
                go.Scatter(
                    x=[x0, x0],
                    y=feature.get('yrange', [0, 1]),
                )
            )
        # ---------------------------------------------
        # CURRENT TRAJECTORY MARKER
        # ---------------------------------------------
        if len(current_marker_indices) > 0:
            # =====================================================
            # PRIMARY AGENT
            # =====================================================
            frame_updates.append(
                go.Scatter(
                    x=[df["x_pos_lowpass"].iloc[i]],
                    y=[df["y_pos_lowpass"].iloc[i]],
                )
            )
            # =====================================================
            # CONFEDERATE
            # =====================================================
            frame_updates.append(
                go.Scatter(
                    x=[df["c_x_pos_lowpass"].iloc[i]],
                    y=[df["c_y_pos_lowpass"].iloc[i]],
                )
            )

        frames.append(
            go.Frame(
                data=frame_updates,
                traces=cursor_trace_indices + current_marker_indices,
                name=str(i)
            )
        )
    fig.frames = frames

    # =========================================================
    # SLIDER
    # =========================================================

    sliders = [{
        "steps": [
            {
                "method": "animate",
                "label": str(i),
                "args": [
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 0,
                            "redraw": False
                        },
                        "transition": {
                            "duration": 0
                        }
                    }
                ]
            }
            for i in range(len(frames))
        ],

        "x": 0.1,
        "y": -0.05,
        "len": 0.85,
    }]

    # =========================================================
    # PLAY / PAUSE
    # =========================================================

    updatemenus = [{
        "type": "buttons",
        "direction": "left",
        "x": 0.1,
        "y": -0.12,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "fromcurrent": True,
                        "frame": {
                            "duration": playback_ms,
                            "redraw": False
                        },
                        "transition": {
                            "duration": 0
                        }
                    }
                ]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 0,
                            "redraw": False
                        }
                    }
                ]
            }
        ]
    }]

    # =========================================================
    # TITLE
    # =========================================================

    if fig_title is None:
        if 'trial_id' in df.columns:
            fig_title = f"Trial {df['trial_id'].iloc[0]}"
        else:
            fig_title = "Trajectory Playback"

    # =========================================================
    # LAYOUT
    # =========================================================

    fig.update_layout(
        title=fig_title,
        height=total_height,
        width=total_width,
        sliders=sliders,
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(
            groupclick="toggleitem"
        ),
    )

    # =========================================================
    # SHOW
    # =========================================================
    fig.show(renderer="browser")