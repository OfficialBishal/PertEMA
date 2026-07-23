"""Command-line interface for PertEMA.

Score a predictor's outputs for per-prediction reliability. Reads a long-format predictions CSV, featurizes it
against the bundled reference, and reports for each perturbation a reliability score, a calibrated error, a
reliability band, and a 90 percent conformal interval.

    pertema score predictions.csv
    pertema score - < predictions.csv --json
    pertema example
"""
import argparse
import csv
import json
import os
import sys

from . import __version__, default_featurizer, default_model, parse_long_csv


def _score_text(csv_text, src_default, dst_default):
    preds, ingest_report = parse_long_csv(csv_text, src_default, dst_default)
    if not preds:
        raise SystemExit("pertema: no valid perturbations parsed from the CSV")
    feat = default_featurizer()
    x, syms, feat_report = feat.featurize(preds)
    if len(x) == 0:
        raise SystemExit("pertema: no perturbations could be featurized (check gene names and contexts)")
    model = default_model()
    scored = model.score(x)
    explained = model.explain(x)
    return syms, scored, explained, feat_report


def _rows(syms, scored, explained):
    for i, gene in enumerate(syms):
        drivers = explained[i].get("top_drivers") or []
        top = drivers[0] if drivers else None
        yield {
            "perturbed_gene": gene,
            "reliability": round(float(scored["reliability"][i]), 6),
            "calibrated_error": round(float(scored["calibrated_error"][i]), 6),
            "band": scored["band"][i],
            "conformal_lo": round(float(scored["conformal_lo"][i]), 6),
            "conformal_hi": round(float(scored["conformal_hi"][i]), 6),
            "top_driver": (top["group"] + " (" + top["direction"] + ")") if top else "",
        }


def _emit(syms, scored, explained, feat_report, output, as_json):
    rows = list(_rows(syms, scored, explained))
    out = open(output, "w", newline="") if output else sys.stdout
    try:
        if as_json:
            json.dump(
                {
                    "model_version": scored["model_version"],
                    "calibration_coverage": scored["coverage"],
                    "mapped_fraction": feat_report["mapped_fraction"],
                    "results": rows,
                },
                out,
                indent=2,
            )
            out.write("\n")
        else:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if output:
            out.close()
    print(
        "scored {n} perturbations, mapped {pct:.0f}% of genes, model {ver}".format(
            n=len(syms), pct=feat_report["mapped_fraction"] * 100, ver=scored["model_version"]
        ),
        file=sys.stderr,
    )


def _example_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "app", "examples", "example_predictions.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pertema",
        description="Post-hoc reliability scoring for single-cell perturbation predictors.",
    )
    parser.add_argument("--version", action="version", version="pertema {0}".format(__version__))
    sub = parser.add_subparsers(dest="command")

    score = sub.add_parser("score", help="score a long-format predictions CSV")
    score.add_argument("csv", help="path to the CSV, or - to read from stdin")
    score.add_argument("-o", "--output", help="write results here (default stdout)")
    score.add_argument("--json", action="store_true", help="emit JSON instead of CSV")
    score.add_argument("--src-default", default="Rest", help="source context when the CSV lacks one")
    score.add_argument("--dst-default", default="Stim48hr", help="destination context when the CSV lacks one")

    example = sub.add_parser("example", help="score the bundled example predictions")
    example.add_argument("-o", "--output")
    example.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "score":
            text = sys.stdin.read() if args.csv == "-" else open(args.csv).read()
            syms, scored, explained, feat_report = _score_text(text, args.src_default, args.dst_default)
            _emit(syms, scored, explained, feat_report, args.output, args.json)
            return 0
        if args.command == "example":
            text = open(_example_path()).read()
            syms, scored, explained, feat_report = _score_text(text, "Rest", "Stim48hr")
            _emit(syms, scored, explained, feat_report, args.output, args.json)
            return 0
    except BrokenPipeError:
        # a downstream reader closed the pipe early (for example, piping to head); exit quietly
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
