"""Audit bookkeeping only: no model fits or dataset mutations."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("text_audit", Path(__file__).parents[1] / "scripts/investigate_text_discrepancies.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_reviewed_categories_are_disjoint_and_cover_37():
    ids = [i for values in audit.GROUPS.values() for i in values]
    assert len(ids) == len(set(ids)) == 37
    assert [len(audit.GROUPS[c]) for c in audit.CATEGORIES] == [21, 8, 8, 0]
    with pytest.raises(ValueError):
        audit.classify(0)


def test_substantive_name_verb_and_unit_changes_are_not_formatting():
    for i in (24, 35, 64, 65, 66, 203, 227, 275):
        assert audit.classify(i) == "substantive_wording"


def test_key_rule_matches_repository_and_retains_punctuation_and_accents():
    from eeg_decoding.data.dataset import sentence_key
    for text in ("  HELLO\tWorld ", "José", "Jose", "1902–1989", "1902-1989", "km²", "km?"):
        assert audit.key(text) == sentence_key(text)
    assert audit.key("José") != audit.key("Jose")
    assert audit.key("1902–1989") != audit.key("1902-1989")
    assert audit.key("  HELLO\tWorld ") == audit.key("hello world")


def test_unique_sentence_counts_do_not_count_subject_repeats_as_sentences():
    rows = [{"sentence_id": "10", "text": "same"}, {"sentence_id": "10", "text": "same"},
            {"sentence_id": "35", "text": "different"}, {"sentence_id": "0", "text": "unaffected"}]
    counts = audit.count_rows(rows, {10: "formatting_only", 35: "substantive_wording"})
    assert counts["formatting_only"] == {"unique_sentences": 1, "observations": 2}
    assert counts["total"] == {"unique_sentences": 2, "observations": 3}
    assert counts["all_observations"] == 4


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        audit.main(tmp_path)
