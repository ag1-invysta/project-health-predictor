from __future__ import annotations
import argparse
import pandas as pd

from .features import compute_features
from .scoring import compute_health
from .explain import add_explanations
from .model import train_on_time_model, predict_on_time

def main() -> None:
    p = argparse.ArgumentParser(prog="health")
    p.add_argument("--snapshots", required=True, help="CSV of project health snapshots")
    p.add_argument("--history", help="Optional labeled history CSV (for ML prediction)")
    p.add_argument("--out", default="health_scored.csv", help="Output CSV path")
    args = p.parse_args()

    df = pd.read_csv(args.snapshots)
    feat = compute_features(df)
    scored = compute_health(feat)
    scored = add_explanations(scored, n=3)

    if args.history:
        hist_raw = pd.read_csv(args.history)
        hist_feat = compute_features(hist_raw)
        model = train_on_time_model(hist_feat)
        scored["p_on_time"] = predict_on_time(model, scored)

    scored.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
