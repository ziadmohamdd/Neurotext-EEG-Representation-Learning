import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('pooling_report',Path(__file__).parents[1]/'scripts/report_temporal_pooling_comparison.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_validation_pool_size_is_measured_from_plan_not_training_bank():
    keys=[str(i) for i in range(298)]
    assert report.validate_pools([{'evaluation_split':'val','candidate_keys':keys}],keys)==298


def test_report_rejects_test_or_different_candidate_pool():
    with pytest.raises(ValueError):
        report.validate_pools([{'evaluation_split':'test','candidate_keys':['a']}],['a'])
    with pytest.raises(ValueError):
        report.validate_pools([{'evaluation_split':'val','candidate_keys':['a','b']}],['a'])
