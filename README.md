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