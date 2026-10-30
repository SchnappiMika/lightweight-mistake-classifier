"""Keep the original time goal and failure denominator in benchmark reports."""
from tools.benchmark_media_batch import summarize


def test_slow_or_failed_item_cannot_be_hidden_by_fast_average():
    rows = [{"total_ms": 1500, "recognized": True, "classified": True},
            {"total_ms": 10, "recognized": False, "classified": False, "error": "bad image"}]
    result = summarize(rows, 1600)
    assert result["attempted"] == 2 and result["failures"] == 1
    assert result["mean_item_ms"] == 755
    assert not result["one_second_requirement_met"]
    assert result["batch_ms_per_item"] == 800


def test_batch_teardown_and_loading_are_included_in_goal():
    assert not summarize([{"total_ms": 100, "classified": True}], 1100)["one_second_requirement_met"]
    assert summarize([{"total_ms": 100, "classified": True}], 200)["one_second_requirement_met"]


def test_empty_batch_cannot_pass_goal():
    result = summarize([], 0)
    assert not result["one_second_requirement_met"]
    assert result["batch_ms_per_item"] is None
