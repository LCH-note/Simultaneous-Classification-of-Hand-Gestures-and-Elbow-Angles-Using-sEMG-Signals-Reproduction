# Simultaneous Classification of Hand Gestures and Elbow Angles

## Week 1 assignment

The environment check is intentionally omitted. Items 2 through 5 are in:

- `week1_assignment.ipynb`: submission notebook
- `src/step2_data_summary.py`: ② data summary
- `src/step3_signal_plot.py`: ③ five-channel signal plot
- `src/step4_angle_comparison.py`: ④ 30°/60° comparison
- `src/step5_class_design.py`: ⑤ one-page class design
- `outputs/week1/step5_class_design.md`: completed one-page class design
- `outputs/week1/figures/`: five-channel and 30°/60° comparison figures
- `outputs/week1/tables/`: reproducible CSV summaries and class design tables

Regenerate all Week 1 outputs from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.week1_assignment
```

Run each assignment item separately:

```powershell
.\.venv\Scripts\python.exe -m src.step2_data_summary
.\.venv\Scripts\python.exe -m src.step3_signal_plot
.\.venv\Scripts\python.exe -m src.step4_angle_comparison
.\.venv\Scripts\python.exe -m src.step5_class_design
```

The analysis uses the 30 trial markers in each CSV, extracts 5,000 samples per
trial, and treats samples 2,000 through 2,999 as the one-second motion interval.
