"""Tests for the pertema command-line interface."""
import json

import pytest

from pertema import cli

SMALL_CSV = "perturbed_gene,gene,predicted_lfc\nIL2,IL2RA,0.83\nIL2,IFNG,-0.21\nSTAT5A,MYC,0.44\n"
BANDS = {"high", "moderate", "low", "very-low"}


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "pertema" in capsys.readouterr().out


def test_score_json(tmp_path, capsys):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(SMALL_CSV)
    assert cli.main(["score", str(csv_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model_version"] == "0.1.0"
    genes = [row["perturbed_gene"] for row in out["results"]]
    assert "IL2" in genes and "STAT5A" in genes
    for row in out["results"]:
        assert row["band"] in BANDS
        assert row["conformal_lo"] <= row["calibrated_error"] <= row["conformal_hi"]


def test_score_csv_row_count(tmp_path, capsys):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(SMALL_CSV)
    assert cli.main(["score", str(csv_path)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("perturbed_gene,reliability,calibrated_error,band")
    assert len(lines) == 3  # header plus IL2 and STAT5A


def test_score_all_unmapped_exits_nonzero(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("perturbed_gene,gene,predicted_lfc\nNOT_A_GENE_XYZ,A,0.5\n")
    with pytest.raises(SystemExit):
        cli.main(["score", str(csv_path)])
