"""Tests for team name normalization and aliases."""

from mlbv1.data.mapping import normalize_team


def test_normalize_team_alt_abbreviations() -> None:
    assert normalize_team("CWS") == "CHW"
    assert normalize_team("KC") == "KCR"
    assert normalize_team("SD") == "SDP"
    assert normalize_team("SF") == "SFG"
    assert normalize_team("TB") == "TBR"
