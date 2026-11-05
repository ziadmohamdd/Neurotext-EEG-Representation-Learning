"""Saved-score display must work from files that a fresh clone contains."""
from pathlib import Path
import runpy
import sys
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/show_accuracy.py'


@pytest.mark.parametrize('study', ['incremental', 'spectral', 'word-order'])
def test_show_accuracy_reads_archived_summaries(study, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['show_accuracy', '--study', study])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    output = capsys.readouterr().out
    assert 'research_results' in output and 'VALIDATION ONLY' in output
