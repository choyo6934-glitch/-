"""실험 기록: 시도한 모든 설정·가설을 results/experiment_log.csv에 덧붙인다(통과 못 한 것 포함)."""
from __future__ import annotations

import csv
import datetime as dt
import json

from config import RESULTS

LOG = RESULTS / "experiment_log.csv"
FIELDS = ["time_utc", "stage", "exp_id", "description", "window", "version", "asset", "cost", "metrics", "decision"]


def log(stage, exp_id, description, window="", version="", asset="", cost="", metrics=None, decision=""):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(time_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), stage=stage,
                        exp_id=exp_id, description=description, window=window, version=version, asset=asset,
                        cost=cost, metrics=json.dumps(metrics or {}, ensure_ascii=False, default=float),
                        decision=decision))
