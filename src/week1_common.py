"""Shared dataset constants and CSV helpers for the Week 1 assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FS = 1_000
TRIAL_SAMPLES = 5_000
MOTION_SLICE = slice(2_000, 3_000)
REST_SLICES = (slice(0, 2_000), slice(3_000, 5_000))
SIGNAL_COLUMNS = [f"bandpass {index}" for index in range(1, 6)]
CHANNEL_NAMES = ["ECR", "ED", "FDS", "Biceps", "Triceps"]
CHANNEL_MUSCLES = [
    "Extensor carpi radialis",
    "Extensor digitorum",
    "Flexor digitorum superficialis",
    "Biceps brachii",
    "Triceps brachii",
]
GESTURE_NAMES = {
    "fist": "Fist",
    "open": "Open hand",
    "pro": "Pronation",
    "extor": "Supination",
}
FILE_PATTERN = re.compile(
    r"(?P<participant>[a-h])_(?P<angle>30|60) "
    r"(?P<gesture>extor|fist|open|pro)\.csv$"
)


@dataclass(frozen=True)
class FileLabel:
    participant: str
    angle: int
    gesture: str

    @property
    def class_name(self) -> str:
        gesture = GESTURE_NAMES[self.gesture].lower().replace(" ", "_")
        return f"{self.angle}_{gesture}"


def parse_label(path: Path) -> FileLabel:
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected dataset filename: {path.name}")
    return FileLabel(
        participant=match.group("participant"),
        angle=int(match.group("angle")),
        gesture=match.group("gesture"),
    )


def trial_starts(frame: pd.DataFrame) -> np.ndarray:
    starts = np.flatnonzero(frame["mark"].to_numpy() != 0)
    return starts[starts + TRIAL_SAMPLES <= len(frame)]


def load_signal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=SIGNAL_COLUMNS + ["time", "onset", "mark"])


def extract_trial(path: Path, trial_number: int = 1) -> np.ndarray:
    frame = load_signal(path)
    starts = trial_starts(frame)
    if trial_number < 1 or trial_number > len(starts):
        raise IndexError(
            f"trial_number must be between 1 and {len(starts)}, got {trial_number}"
        )
    start = int(starts[trial_number - 1])
    return frame.loc[
        start : start + TRIAL_SAMPLES - 1, SIGNAL_COLUMNS
    ].to_numpy()


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{decimals}f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
