"""Write a SYNTHETIC SST validation run to disk.

Examples (run from the project folder):
    python scripts/generate_synthetic.py --revision A --out data/SYNTHETIC_SST_revA_baseline.csv
    python scripts/generate_synthetic.py --revision B --seed 43 --out data/SYNTHETIC_SST_revB_candidate.csv
    python scripts/generate_synthetic.py --revision A --sensor-fault --out data/SYNTHETIC_SST_revA_sensorfault.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow "python scripts/..." without installing

from powerops.synthetic import REVISIONS, SyntheticConfig, synthetic_csv_bytes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SYNTHETIC multi-port SST validation run (CSV).")
    parser.add_argument("--revision", choices=sorted(REVISIONS), default="A",
                        help="A = baseline, B = candidate with regressions, C = fixes B but slow battery pickup (default A)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--sensor-fault", action="store_true", help="add a +3 kW offset to the battery power reading")
    parser.add_argument("--out", type=Path, default=Path("data/SYNTHETIC_SST_run.csv"))
    args = parser.parse_args()

    cfg = SyntheticConfig(revision=args.revision, seed=args.seed,
                          battery_sensor_offset_kw=3.0 if args.sensor_fault else 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(synthetic_csv_bytes(cfg))
    print(f"Wrote SYNTHETIC {REVISIONS[args.revision].label} run to {args.out}.")


if __name__ == "__main__":
    main()
