import os
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

_DESCRIPTION =  "After running `offsets.py` to calculate per-trial offsets, " + \
                "Plot them in a boxplot to analyze their general distribution."

def natural_sorting_by_key(series: pd.Series) -> pd.Series:
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(s):
        return tuple(convert(c) for c in re.split(r'(\d+)', str(s)))
    return series.map(alphanum_key)

def main(pid:str, src_dir:str, offset_src:str, ts_col:str='unix_ms', method:str='boxplot'):

    df = pd.read_csv(os.path.join(src_dir, offset_src))
    df['participant_id'] = pid
    df = df.sort_values(by=["participant_id", "trial_id", "overlap_counter"], key=natural_sorting_by_key)
    
    # Generate the plot
    plt.figure(figsize=(2, 5))  # scale width with N
    if method == 'boxplot':
        sns.boxplot(
            data=df,
            x="participant_id",   # horizontal orientation: y is category, x is value
            y="offset_eeg-gaze",
            #hue='trial_id',
            orient="v",
            showfliers=False,  # hide outlier dots (since we'll show raw data)
            width=0.6,
            boxprops=dict(alpha=.35)
        )
    elif method == 'violin':
        sns.violinplot(
            data=df,
            x="participant_id",
            y="offset_eeg-gaze",
            #hue='trial_id',
            orient="v",
            inner=None,
            cut=0,
            dodge=True,
            linewidth=1,
            alpha=0.35
        )
    # Overlay jittered raw points
    ax = sns.stripplot(
        data=df,
        x="participant_id",
        y="offset_eeg-gaze",
        #hue='trial_id',
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
    #plt.axhline(y=0, c='black', alpha=0.5)
    plt.axhline(y=df['offset_eeg-gaze'].median(), c='blue', alpha=0.25)
    # Avoid duplicate legends from boxplot + stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Adjust the layout
    plt.tight_layout()
    plt.show()
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument('pid', help="The participant ID", type=str)
    parser.add_argument('src_dir', help="The relative path to the participant's directory.", type=str)
    parser.add_argument('-os', '--offset_src', help="The expected filename for offset files", type=str, default='offsets.csv')
    parser.add_argument('-tc', '--timestamp_column', help="The timestamp column name", type=str, default="unix_ms")
    parser.add_argument('-m', '--method', help="What plot type should we use?", type=str, choices=['boxplot', 'violin'], default='boxplot')
    args = parser.parse_args()
    main(
        args.pid,
        args.src_dir,
        args.offset_src,
        ts_col = args.timestamp_column,
        method = args.method
    )