import os
import argparse
import pandas as pd

from RecordMuse.processing import filter, normalize
from RecordMuse.analysis import psd, validate

def main(
        rest_src:str, 
        sim_src:str, 
        ts_col:str='lsl_unix_ts', 
        start_buffer=5, 
        end_buffer=5,
        v:bool=True,
        convert_timestamp:bool=True ):
    # Validations
    assert os.path.exists(rest_src), "Rest state EEG file not found"
    assert os.path.exists(sim_src), "Simulation EEG file not found"

    # First filter both the rest and simulation eeg
    frest_src = filter.filter_eeg(rest_src, apply_bandpass=True)
    fsim_src = filter.filter_eeg(sim_src, apply_bandpass=True)
    print(frest_src, fsim_src)

    # Now normalize
    normed_eeg_src = normalize.normalize(frest_src, fsim_src, ts_col=ts_col, start_buffer=start_buffer, end_buffer=end_buffer, validate=v)
    
    # PSD calculation
    psd_src = psd.calculate_psd(normed_eeg_src)

    # Validate
    validate.validate(os.path.dirname(psd_src), ts_col=ts_col, with_lines=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse the EEG of a participant's data collection session")
    parser.add_argument('rest_eeg_src', help="Relative path to the rest state EEG", type=str)
    parser.add_argument('sim_eeg_src', help='Relative path to the simulation EEG', type=str)
    parser.add_argument('-tc', '--timestamp_column', help='The column name representing timestamps', type=str, default='unix_ms')
    parser.add_argument('-sb', '--start_buffer', help='The amount of time removed from the start of the rest state EEG during normalization', default=5)
    parser.add_argument('-eb', '--end_buffer', help='The amount of time removed from the end of the rest state EEG during normalization', default=5)
    parser.add_argument('-v', '--validate', help='After normalization, should we validate?', action='store_true')
    args = parser.parse_args()

    # Validate that sb and eb are numbers
    start_buffer = float(args.start_buffer)
    end_buffer = float(args.end_buffer)

    # Run
    main(
        args.rest_eeg_src, 
        args.sim_eeg_src, 
        ts_col=args.timestamp_column, 
        start_buffer=start_buffer,
        end_buffer=end_buffer,
        v=args.validate )

