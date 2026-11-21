"""Extract the borderline-case lookup from archived ArNet2 Monte Carlo outputs."""
import argparse
import csv
from pathlib import Path

PATIENTS = ("patient_033", "patient_052")
RADII = (("fixed_radius", "fixed"), ("scaled_radius", "adaptive"))


def load(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    primary = {(row["patient"], row["method"]): row for row in load(args.thresholds / "patient_mc_intervals.csv")
               if row["detector"] == "ArNet2"}
    repeat = {row["patient"]: row for row in load(args.thresholds / "independent_mc_patients.csv")}
    fields = ["patient", "detector", "radius", "original_draws_per_record", "original_coverage_pct",
              "original_mc95_low_pct", "original_mc95_high_pct", "repeat_draws_per_record",
              "repeat_coverage_pct", "repeat_mc95_low_pct", "repeat_mc95_high_pct"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as dest:
        write = csv.writer(dest, lineterminator="\n")
        write.writerow(fields)
        for pid in PATIENTS:
            new = repeat[pid]
            for method, prefix in RADII:
                old = primary[(pid, method)]
                assert old["records"] == new["records"] == "1"
                assert old["coverage_pct"] == new[prefix + "_primary_pct"]
                write.writerow([pid, "ArNet2", prefix, old["draws_per_record"],
                                old["coverage_pct"], old["mc95_low_pct"], old["mc95_high_pct"],
                                new["draws_per_record"], new[prefix + "_independent_pct"],
                                new[prefix + "_mc95_low_pct"], new[prefix + "_mc95_high_pct"]])


if __name__ == "__main__":
    main()
