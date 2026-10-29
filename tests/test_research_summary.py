"""Summary selection must not silently choose a favorable repeat."""
import pytest

from scripts.summarize_research import equivalent_report


def test_equal_export_values_are_represented_once_without_mutation():
    early = dict(name='run_01.json', file='metrics/run_01.json', mrr='0.02', seed='42',
                 training_control='gaussian', candidate_pool_sha256='pool')
    later = dict(early, name='run_02.json', file='metrics/run_02.json')
    reports = [later, early]
    assert equivalent_report(reports) is early
    assert len(reports)==2 and later['file']=='metrics/run_02.json'


@pytest.mark.parametrize('field', ['mrr','seed','training_control','candidate_pool_sha256'])
def test_differing_repeats_require_review_instead_of_score_selection(field):
    first = dict(name='a.json', file='a.json', mrr='0.02', seed='42',
                 training_control='gaussian', candidate_pool_sha256='pool')
    second = dict(first, name='b.json', file='b.json')
    second[field] = '0.99' if field=='mrr' else 'different'
    with pytest.raises(ValueError,match='explicit run review required'):
        equivalent_report([first,second])


def test_missing_condition_is_not_fabricated():
    with pytest.raises(ValueError,match='missing'):
        equivalent_report([])
