# MIT-BIH Arrhythmia Dataset

This folder is intentionally empty.

ECG data is downloaded automatically at runtime via the `wfdb` Python library,
which streams records directly from PhysioNet (https://physionet.org).

No manual download is required. The library caches downloaded files locally
in this directory after the first run.

## Records Used

| Record | Rhythm Type           | Usage    |
|--------|-----------------------|----------|
| 100    | Normal sinus          | Train    |
| 101    | Normal sinus          | Train    |
| 103    | Normal sinus          | Train    |
| 105    | Left bundle branch    | Train    |
| 106    | PVCs (some)           | Train    |
| 108    | PVCs (frequent)       | Test     |
| 200    | Mixed arrhythmia      | Test     |
| 207    | Complete heart block  | Test     |

## Citation

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)