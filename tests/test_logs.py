import json
from types import SimpleNamespace

import pytest

from llmkit.loaders import DocumentError
from llmkit.logs import (
    LOG_COLUMNS,
    events_to_frame,
    parse_log,
    process_log_file,
    recover_unparsed,
)

SAMPLE = """2026-03-14 10:00:01 INFO service started
2026-03-14T10:00:02.250Z [ERROR] payment failed
    Traceback line one
    Traceback line two
garbled line without a timestamp
2026-03-14 10:00:05,500 WARN disk at 91%
"""

RECOVERED = {
    "timestamp": None,
    "level": "INFO",
    "message": "garbled line without a timestamp",
}


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, total_s=0.1, cost_usd=0.0)


async def no_sleep(delay):
    return None


def test_parse_log_reads_levels_timestamps_and_continuations():
    result = parse_log(SAMPLE)
    assert [p.line_no for p in result.parsed] == [1, 2, 6]
    assert [p.event.level.value for p in result.parsed] == ["INFO", "ERROR", "WARNING"]
    assert result.parsed[1].event.timestamp.microsecond == 250000
    assert result.parsed[2].event.timestamp.microsecond == 500000
    assert "Traceback line one" in result.parsed[1].event.message
    assert "Traceback line two" in result.parsed[1].event.message
    assert result.unparsed == [(5, "garbled line without a timestamp")]


def test_indented_line_after_garbage_is_not_attached():
    result = parse_log("2026-03-14 10:00:01 INFO ok\ngarbled\n  indented")
    assert len(result.parsed) == 1
    assert result.parsed[0].event.message == "ok"
    assert result.unparsed == [(2, "garbled"), (3, "indented")]


def test_unknown_level_goes_to_unparsed():
    result = parse_log("2026-03-14 10:00:01 BANANA hello")
    assert result.parsed == []
    assert len(result.unparsed) == 1


def test_blank_and_empty_text():
    assert parse_log("").parsed == []
    assert parse_log("\n  \n").unparsed == []


async def test_recover_uses_llm_and_tags_source():
    result = parse_log(SAMPLE)
    primary = FakeClient(json.dumps(RECOVERED))
    result = await recover_unparsed(result, primary, FakeClient(), sleep=no_sleep)
    assert result.recovered == 1
    assert result.unparsed == []
    assert [p.line_no for p in result.parsed] == [1, 2, 5, 6]
    assert [p.source for p in result.parsed] == ["regex", "regex", "llm", "regex"]


async def test_failed_recovery_stays_unparsed():
    result = parse_log(SAMPLE)
    result = await recover_unparsed(
        result, FakeClient("not json"), FakeClient("not json"), sleep=no_sleep
    )
    assert result.recovered == 0
    assert result.unparsed == [(5, "garbled line without a timestamp")]


async def test_max_lines_limits_llm_calls():
    result = parse_log("garbled one\ngarbled two\ngarbled three")
    primary = FakeClient(json.dumps(RECOVERED))
    result = await recover_unparsed(result, primary, FakeClient(), max_lines=1, sleep=no_sleep)
    assert primary.calls == 1
    assert result.recovered == 1
    assert len(result.unparsed) == 2


def test_events_to_frame_is_ordered_and_flat():
    frame = events_to_frame(parse_log(SAMPLE))
    assert list(frame.columns) == LOG_COLUMNS
    assert list(frame["line_no"]) == [1, 2, 6]
    assert list(frame["level"]) == ["INFO", "ERROR", "WARNING"]


def test_events_to_frame_empty_keeps_columns():
    frame = events_to_frame(parse_log(""))
    assert frame.empty
    assert list(frame.columns) == LOG_COLUMNS


async def test_process_log_file_end_to_end(tmp_path):
    path = tmp_path / "app.log"
    path.write_text(SAMPLE)
    result = await process_log_file(
        path, FakeClient(json.dumps(RECOVERED)), FakeClient(), sleep=no_sleep
    )
    assert len(result.parsed) == 4
    assert result.recovered == 1


async def test_process_log_file_rejects_empty_file(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("  \n")
    with pytest.raises(DocumentError, match="empty"):
        await process_log_file(path, FakeClient(), FakeClient())
