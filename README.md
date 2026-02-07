# Pedestrian Encounters

## Dependencies

- **RecordMuse** ([Github](https://github.com/SimpleDevs-Tools/RecordMuse))

## Notes

- Each participant is likely to contain several "calibration" files - specifically, **eight** (8) of them. These occur prior to each trial.
- There are only 6 trials + an acclimation session. So there is:
    1. An unofficial "acclimation" session (`calibration_test.csv`)
    2. 6 trial calibrations (`calibration_test_1.csv` - `caliration_test_6.csv`)
    3. One final closing calibration to mark the end (`calibration_test_7.csv`)
    We only really need the ones in #2 (`calibration_test_1.csv` - `caliration_test_6.csv`)
- The actual trial start and ends are extracted from `eye.csv`. For some reason, we thought it was best to condense all that into `eye.csv`
- We record a rest-state EEG as well.

## Theoretical Processing

For each participant, we must do the following:

1. Process the EEG data, such as normalizing the data and filtering out unwanted signals (`RecordMuse/`)
2. Determine and match up trials with their calibration sessions.
3. For each trial, splice the data and segment them

## List of commands:

For each participant:

You might have the following:

- `calibration_test_<#>.csv` or `calibration_<#>.csv`
    - Each of these represent calibration files generated at the start of each trial.
    - Early samples might have only 7. Later ones will likely have 8 of them. Though we only have 6 trials, there are 7 because there's a final calibration session at the end of all the trials.
    - If there are 7, then there is no calibration session at the start of the entire session. if there are 8, then there is an additional calibration session at the start of the entire session.
    - You really only need to care about the first 6 (if you have 7 calibration files) or the middle 6 (if you have 8).
- `eeg-rest.csv` and `eeg-vr.csv`: The EEG data, if you worked with Mind Monitor (very likely in older participant sessions). 
    - You'll have to convert these to adhere to BlueMuse's format, if you haven't already
- `eye.csv` = the eye data measured from the participant.
    - Spans the entire session
    - Also includes the participant's position and head orientation
- `pedestrians.csv` = The pedestrian movement data. Early participant sessions will only distinguish them by `id` and will all have the same `Label`. This is rectified in later data sessions, but be forewarned.

The general steps you must perform during post-processing is thus:

1. Convert your EEG data into BlueMuse's format, if needed.
    ```bash
    python RecordMuse/processing/convert.py [path to either `eeg-rest.csv` or `eeg-vr.csv`]
    ```
2. Process your EEG data. You first need to:
    1. Filter your EEG:
        ```bash
        python RecordMuse/processing/filter.py [path to your rest or vr eeg `.csv`] -b -v
        ```
    2. Normalize your EEG using both the VR and rest EEG:
        ```bash
        python RecordMuse/processing/normalize.py [path to rest EEG] [path to VR eeg] [-tc] [-sb] [-eb] [-v]
        ```
    3. Calculate the bandpowers of your EEG data
        ```bash
        python RecordMuse/analysis/psd.py [path to filtered, normalized EEG]
        ```
3. Interpret your trial timings and sessions.
    ```bash
    python trials.py [root data directory of participant] [-es] [-tc]
    ```
4. Estimate the offsets of your participant, based on the EEG and eye data
    ```bash
    python offsets.py [root data directory of participant] [-ts] [-es] [-ees] [-tc] [-sb] [-eb] [-s]
    ```
5. After calculating the offsets, you must separate the original data into their own trials.
    ```bash
    python separate.py [src directory to participant] [-o] [-t] [-eeg] [-ey] [-p] [-tc]
    ```
6. You can proceed to generate events from all the VR-generated data
    ```bash
    python events.py [src dirctory to participant]
    ```
