import os
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

_DESCRIPTION =  "After running `offsets.py` to calculate per-trial offsets, " + \
                "Plot them in a boxplot to analyze their general distribution. " + \
                "This script will automatically detect all `offsets.csv` " + \
                "available within the provided `src_dir`."

def natural_sorting_by_key(series: pd.Series) -> pd.Series:
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(s):
        return tuple(convert(c) for c in re.split(r'(\d+)', str(s)))
    return series.map(alphanum_key)

def analyze_offsets(src_dir:str, offset_src:str, ts_col:str='unix_ms', trial_hue:bool=False, method:str='boxplot', show:bool=False):

    # Find all `offset_src` files in `src_dir`
    offset_dfs = []
    for root, dirs, files in os.walk(src_dir):
        if offset_src in files:
            offset_dfs.append(pd.read_csv(os.path.join(root, offset_src)))
    
    # Derive the main df
    df = pd.concat(offset_dfs, ignore_index=True)
    df = df.sort_values(by=["pid", "trial_id", "overlap_counter"], key=natural_sorting_by_key)
    
    # Determine hues
    hue, hue_order = 'trial_id',  sorted(df['trial_id'].unique())
    if not trial_hue:
        hue, hue_order = None, None

    # Generate the plot
    plt.figure(figsize=(2*len(df.pid.unique()), 5))  # scale width with N
    if method == 'boxplot':
        sns.boxplot(
            data=df,
            x="pid",   # horizontal orientation: y is category, x is value
            y="offset_eeg-gaze",
            hue=hue,
            hue_order=hue_order,
            orient="v",
            showfliers=False,  # hide outlier dots (since we'll show raw data)
            width=0.6,
            boxprops=dict(alpha=.35)
        )
    elif method == 'violin':
        pid_medians = (
            df.groupby("pid")["offset_eeg-gaze"]
            .median()
        )
        pid_means = (
            df.groupby("pid")["offset_eeg-gaze"]
            .mean()
        )
        sns.violinplot(
            data=df,
            x="pid",
            y="offset_eeg-gaze",
            hue=hue,
            hue_order=hue_order,
            orient="v",
            inner=None,
            cut=0,
            dodge=True,
            linewidth=1,
            alpha=0.35
        )
        ax = plt.gca()
        xticks = ax.get_xticks()
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        pid_to_x = dict(zip(xlabels, xticks))
        half_width = 0.25
        for pid, median in pid_medians.items():
            x = pid_to_x[str(pid)]
            ax.hlines(
                y=median,
                xmin=x - half_width,
                xmax=x + half_width,
                colors="blue",
                linewidth=2,
                alpha=0.8
            )
        for pid, mean in pid_means.items():
            x = pid_to_x[str(pid)]
            ax.hlines(
                y=mean,
                xmin=x - half_width,
                xmax=x + half_width,
                colors="red",
                linewidth=2,
                alpha=0.8
            )
    # Overlay jittered raw points
    ax = sns.stripplot(
        data=df,
        x="pid",
        y="offset_eeg-gaze",
        hue=hue,
        hue_order=hue_order,
        dodge=True,
        orient="v",
        alpha=1.0,
        jitter=0.02,
        linewidth=1,
        edgecolor='gray',
        size=4,
    )
        
    # Other plot stuff
    plt.title("Diff Distributions per Session")
    plt.xlabel("Participant ID")
    plt.ylabel("Offset (EEG - VR, ms)")
    # Add lines to indicate the 0-diff line
    plt.axhline(y=0, c='black', alpha=0.25)
    #plt.axhline(y=df['offset_eeg-gaze'].median(), c='blue', alpha=0.25)
    # Avoid duplicate legends from boxplot + stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Adjust the layout
    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, 'offsets.png'), bbox_inches='tight', dpi=300)
    if show:    plt.show()
    else:       plt.close()
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument('src_dir', help="The relative path to where all participant folders are stored", type=str)
    parser.add_argument('-os', '--offset_src', help="The expected filename for offset files", type=str, default='offsets.csv')
    parser.add_argument('-tc', '--timestamp_column', help="The timestamp column name", type=str, default="unix_ms")
    parser.add_argument('-th', '--trial_hue', help="Should we hue based on `trial_id`?", action='store_true')
    parser.add_argument('-m', '--method', help="What plot type should we use?", type=str, choices=['boxplot', 'violin'], default='boxplot')
    parser.add_argument('-s', '--show', help='If toggled, will show the offset plot on screen prior to closing', action='store_true')
    args = parser.parse_args()
    analyze_offsets(
        args.src_dir,
        args.offset_src,
        ts_col = args.timestamp_column,
        trial_hue = args.trial_hue,
        method = args.method,
        show = args.show
    )