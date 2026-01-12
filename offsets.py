import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

_DESCRIPTION =  "Given a set of trials (first derived from `trials.csv`), " + \
                "attempt to calculate the offset based on peaks and valleys. " + \
                "Whether this is a manual or automated process is toggable. " + \
                "If this is an automated process, then signal smoothing will occur. " + \
                "We can toggle either only manual (default), automated, or both."

class PointSelector:
    def __init__(self, 
                 data,           # [(x, y, label, color), ...]
                 which_plot=0,
                 figsize=None,
                 title: str = None,
                 previous=None,
                 vlines=None):

        # Assertions
        if previous is not None:
            assert len(data) == len(previous), (
                "If previous is provided, must match number of subplots."
            )

        # Cache data
        self.data = [{
            'x': np.asarray(d[0]),
            'y': np.asarray(d[1]),
            'label': d[2],
            'color': d[3]
        } for d in data]

        self.figsize = figsize or (10, len(self.data) * 3)
        self.title = title
        self.previous = previous

        # Selection state
        self.which_plot = which_plot
        self.idx = 0
        self.selected = None

        # Create plots
        self.fig, self.axes = plt.subplots(
            len(self.data), 1,
            figsize=self.figsize,
            sharex=True
        )

        # Normalize axes to list
        if len(self.data) == 1:
            self.axes = [self.axes]

        for i, ax in enumerate(self.axes):
            x = self.data[i]['x']
            y = self.data[i]['y']

            ax.plot(
                x, y, "-o",
                color=self.data[i]['color'],
                label=self.data[i]['label'],
                alpha=0.3,
                markersize=1
            )
            ax.legend(loc="upper left")

            # Static vertical markers (if provided)
            if vlines is not None:
                ax_transform = mtransforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )
                for xv, label in vlines:
                    ax.axvline(x=xv, color="black", alpha=0.5)
                    ax.text(
                        xv, 0.02, label,
                        transform=ax_transform,
                        rotation=90,
                        ha="left",
                        va="bottom",
                        fontsize=6,
                        rotation_mode="anchor"
                    )

            # Previous selections
            if self.previous is not None and len(self.previous[i]) > 0:
                _, px, py = zip(*self.previous[i])
                ax.plot(
                    px, py, "x",
                    color="black",
                    markersize=8,
                    label="Previous",
                    zorder=2
                )
                ax.text(
                    1.02, 1.0,
                    self.format_previous(i),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    family="monospace"
                )

        # Movable red marker (active plot only)
        ax = self.axes[self.which_plot]
        x0 = self.data[self.which_plot]['x'][self.idx]
        y0 = self.data[self.which_plot]['y'][self.idx]
        self.marker, = ax.plot(x0, y0, "ro", markersize=10, zorder=3)

        # Cross-plot vertical cursors
        self.vcursor_lines = []
        for ax in self.axes:
            line = ax.axvline(
                x=x0,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.4,
                zorder=1
            )
            self.vcursor_lines.append(line)

        # Event connections
        self.cid = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.mid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.update()

    def format_previous(self, i: int = 0):
        if not self.previous:
            return "Previous selections:\n(none)"
        prev = self.previous[i]
        lines = ["Previous selections:"]
        for j, (_, x, y) in enumerate(prev, 1):
            lines.append(f"{j:>2}: ({x:.3f}, {y:.3f})")
        return "\n".join(lines)

    def update(self):
        x_sel = self.data[self.which_plot]['x'][self.idx]
        y_sel = self.data[self.which_plot]['y'][self.idx]

        # Update red marker
        self.marker.set_data([x_sel], [y_sel])

        # Update vertical alignment lines
        for i, line in enumerate(self.vcursor_lines):
            line.set_xdata([x_sel, x_sel])
            line.set_visible(i != self.which_plot)

        title_text = (
            f"Index {self.idx} | x={x_sel:.3f}, y={y_sel:.3f}"
        )
        if self.title is not None:
            title_text = self.title

        self.axes[self.which_plot].set_title(
            title_text + "\n← / → move | ENTER: select | ESC: skip"
        )

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.update()

        elif event.key == "right":
            self.idx = min(
                len(self.data[self.which_plot]['x']) - 1,
                self.idx + 1
            )
            self.update()

        elif event.key == "enter":
            self.selected = (
                self.idx,
                self.data[self.which_plot]['x'][self.idx],
                self.data[self.which_plot]['y'][self.idx]
            )
            self.close()

        elif event.key == "escape":
            self.selected = None
            self.close()

    def on_click(self, event):
        if event.inaxes != self.axes[self.which_plot]:
            return
        if event.xdata is None:
            return

        self.idx = np.argmin(
            np.abs(self.data[self.which_plot]['x'] - event.xdata)
        )
        self.update()

    def close(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.fig.canvas.mpl_disconnect(self.mid)
        plt.close(self.fig)


def calculate_offsets(
        src_dir:str, 
        trial_src:str, 
        eye_src:str, 
        eeg_src, 
        ts_col:str='unix_ms', 
        start_buffer:float = 5000,
        end_buffer:float = 500,
        method:str="manual" ):

    # Read trials, eye, and eeg data
    trials = pd.read_csv(os.path.join(src_dir, trial_src))
    eye = pd.read_csv(os.path.join(src_dir, eye_src))
    eeg = pd.read_csv(os.path.join(src_dir, eeg_src))
    assert ts_col in eye.columns, f"The expected timestamp column \"{ts_col}\" is not present in \"{eye_src}\""
    assert ts_col in eeg.columns, f"The expected timestamp column \"{ts_col}\" is not present in \"{eeg_src}\""
    eye = eye.sort_values(by=ts_col)
    eeg = eeg.sort_values(by=ts_col)

    # Iterate through trials, get their respective calibration data, 
    #   slice the eye and eeg data based on the calibration start and end times, and
    #   select representative points manually. Then save the offsets inside the trial directories as `offsets.csv`.
    offsets_dfs = []
    for i, trial in trials.iterrows():
        # Get the calibration data
        trial_id = trial['trial_id']
        cdf = pd.read_csv(os.path.join(src_dir, str(trial_id), 'calibration.csv'))
        start_unix_ms = cdf.loc[cdf["event"] == "Start"].iloc[0]['unix_ms'] + start_buffer
        end_unix_ms = cdf.loc[cdf["event"] == "End"].iloc[0]['unix_ms'] - end_buffer
        overlap_rows = cdf[cdf['event'] == 'Overlap']

        # Extract the individual data for this calibration
        trial_eye = eye[eye[ts_col].between(start_unix_ms, end_unix_ms)]
        trial_eeg = eeg[eeg[ts_col].between(start_unix_ms, end_unix_ms)]
        gx = trial_eye[ts_col].to_list()
        gy = trial_eye['gaze_target_screen_pos_y'].to_list()
        ex = trial_eeg[ts_col].to_list()
        ey = trial_eeg['TP9'].to_list()
        if (len(gx) == 0 or len(gy) == 0 or len(ex) == 0 or len(ey) == 0):
            print("ERROR: No data associated with calibration step. Skipping")
            continue
        data = ((gx, gy, 'Gaze Y', 'red'), (ex, ey, 'TP9', 'blue'))
        # Vertical lines represent when the overlaps occur during the calibration stage
        vlines = [(row['unix_ms'], f"Start {row['overlap_counter']}") for row_index, row in overlap_rows.iterrows()]
        
        # Iterate through overlaps
        overlaps = []
        eegs = []
        gazes = []
        for _, overlap in overlap_rows.iterrows():
            # Get overlap coiunter
            overlap_counter = overlap['overlap_counter']
            title = f"Overlap {overlap_counter}"
            # Define our selector for gaze
            selector = PointSelector(data, which_plot=0, vlines=vlines, previous=(gazes, eegs), title=title)
            plt.show()
            gaze_selection = selector.selected
            if gaze_selection is None: continue
            # Define our selector for eeg
            selector = PointSelector(data, which_plot=1, vlines=vlines, previous=(gazes+[gaze_selection], eegs), title=title) 
            plt.show()
            eeg_selection = selector.selected
            if eeg_selection is None: continue
            # Add to our selections
            overlaps.append(overlap_counter)
            gazes.append(gaze_selection)
            eegs.append(eeg_selection)
        # If any of the lists are empty, then this is an invalid calibration. We need to indicate that this is the case.
        if len(overlaps) == 0:
            # No calibrations allowed. Reject
            print("No calibration blinks, and thus no offsets, detected in this calibration trial...")
            continue
        agg_selections = list(zip(overlaps, gazes, eegs))
        agg_entries = [{
            'overlap_counter':s[0], 
            'gaze_unix_ms':int(s[1][1]), 
            'eeg_unix_ms':int(s[2][1])} for s in agg_selections]
        # Calculate offsets, save the offsets within this trial
        df = pd.DataFrame(agg_entries)
        df['trial_id'] = trial['trial_id']
        df['offset_eeg-gaze'] = df['eeg_unix_ms'] - df['gaze_unix_ms']
        offsets_dfs.append(df)
        print(f"Successfully collected offsets for Trial {trial_id}")

    # Concat all offsets, then save
    if len(offsets_dfs) == 0:
        print("ERROR: No offsets measured for this participant. This is an invalid participant")
        return None
    df = pd.concat(offsets_dfs)
    outpath = os.path.join(src_dir, 'offsets.csv')
    df.to_csv(outpath, index=False)
    print(f"Saved all offsets!")
    return outpath


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument('src_dir', help="The directory of the participant. Should contain `vr/` and `eeg/` subdirectories.", type=str)
    parser.add_argument('-ts', '--trial_src', help="The filepath relative to `src_dr` that references the participant's trial listings", type=str, default="trials.csv")
    parser.add_argument('-es', '--eye_src', help="The filepath relative to `src_dir` that references the participant's eye data", type=str, default='eye.csv')
    parser.add_argument('-ees', '--eeg_src', help="The filepath relative to `src_dr>` that references the participant's EEG data, usually after filtering and normalization", type=str, default="EEG_filtered_normalized.csv")
    parser.add_argument('-tc', '--timestamp_column', help='The timestamp column of choice', type=str, default='unix_ms')
    parser.add_argument('-sb', '--start_buffer', help='The amount of time removed from the start of each calibration stage', default=5000)
    parser.add_argument('-eb', '--end_buffer', help="The amount of time removed from the end of each calibration stage.", default=500)
    parser.add_argument('-m', '--method', help="Manual, automated, or both modes.", type=str, choices=["manual", "automated", "both"], default="manual")
    args = parser.parse_args()
    start_buffer = float(args.start_buffer)
    end_buffer = float(args.end_buffer)
    calculate_offsets(
        args.src_dir, 
        args.trial_src, 
        args.eye_src, 
        args.eeg_src, 
        ts_col = args.timestamp_column,
        start_buffer = start_buffer,
        end_buffer = end_buffer,
        method=args.method  )