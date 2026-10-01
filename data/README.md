# Data Requirements & Provenance

This research evaluates multi-agent defensive transitions using the **Metrica Sports Open Tracking Dataset** (Sample Games 1, 2, and 3).

## 1. Automated Download
To download the complete 25 Hz tracking and event telemetry for Games 1–3, run the automated downloader script from the repository root:

```bash
python download_metrica.py
```

This will fetch the raw files directly from the [official Metrica Sports repository](https://github.com/metrica-sports/sample-data):
- **Game 1**: `Sample_Game_1_RawEventsData.csv`, `Sample_Game_1_RawTrackingData_Home_Team.csv`, `Sample_Game_1_RawTrackingData_Away_Team.csv`
- **Game 2**: `Sample_Game_2_RawEventsData.csv`, `Sample_Game_2_RawTrackingData_Home_Team.csv`, `Sample_Game_2_RawTrackingData_Away_Team.csv`
- **Game 3**: `Sample_Game_3_events.json`, `Sample_Game_3_metadata.xml`, `Sample_Game_3_tracking.txt`

## 2. Standalone Offline Replication (No Large Downloads Required)
Because raw 25 Hz tracking files total ~175 MB and take time to parse, this repository ships with the complete pre-computed transition dataset:

`data/batch_analysis_results.json` (47 KB)

This file contains the complete transition event records ($N = 69$ events: 11 conceded goals, 58 non-scoring shots) with:
- Initial observed threat surfaces
- Counterfactual threat-minimizing coordinates $p^*$
- Ghost displacement $\|p^* - p_0\|$ (m)
- Counterfactual threat reduction $\Delta\text{Threat}$ (%)
- Diagnostic triage classifications

All paper figures can be reproduced instantly without downloading the raw tracking telemetry:
```bash
python reproduce_figures.py
```

## 3. Coordinate Conventions & Preprocessing
- **Pitch Dimensions**: Normalized to $105\text{ m} \times 68\text{ m}$.
- **Tracking Rate**: Continuous 25 Hz optical tracking.
- **Velocity Smoothing**: 7-frame Savitzky-Golay filter to eliminate discrete differencing noise.
- **Anonymization**: All player identifiers are anonymized by Metrica Sports (`Player 1`, `Player 2`, etc.).

## 4. Citation & Terms of Use
The Metrica Sports dataset is made available for non-commercial research and educational purposes. When using this data, please cite:
> Metrica Sports (2020). *Sample Tracking and Event Data*. GitHub: https://github.com/metrica-sports/sample-data
