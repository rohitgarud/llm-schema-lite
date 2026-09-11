"""Tests for the accuracy arm's case generator and field-match metric.

The metric decides which adapter wins, so the properties asserted here are the ones that
would silently corrupt a published comparison if they broke: the denominator, how a failed
cell scores, and whether the generated prose actually states every field it grades.
"""

from __future__ import annotations

import random

import pytest

from benchmarking.dspy_adapters.accuracy import (
    AccuracyRow,
    FieldScore,
    aggregate_accuracy,
    flatten,
    normalize,
    score,
)
from benchmarking.dspy_adapters.cases import (
    Address,
    Contact,
    Employment,
    PersonRecord,
    generate,
    render,
)
from benchmarking.dspy_adapters.external import align_entity_lists
from benchmarking.dspy_adapters.outcomes import Outcome


def _record(**overrides) -> PersonRecord:
    base = {
        "name": "Ada Lovelace",
        "age": 36,
        "address": Address(street="12 Baker St", city="London", postcode="NW1 6XE"),
        "employment": Employment(company="Analytical Engines Ltd", role="research lead", years=4),
        "contacts": [Contact(email="ada@example.com", phone="+44 123 4567")],
        "tags": ["mathematics"],
    }
    base.update(overrides)
    return PersonRecord(**base)


class TestFlatten:
    def test_nests_and_indexes(self):
        flat = flatten(_record())
        assert flat["name"] == "Ada Lovelace"
        assert flat["address.city"] == "London"
        assert flat["employment.years"] == 4
        assert flat["contacts[0].email"] == "ada@example.com"
        assert flat["tags[0]"] == "mathematics"

    def test_keeps_none_leaves(self):
        """A correctly-absent optional is a field the model can get right, so it must count."""
        flat = flatten(_record(employment=None))
        assert flat["employment"] is None

    def test_empty_list_has_no_paths(self):
        flat = flatten(_record(contacts=[], tags=[]))
        assert not [k for k in flat if k.startswith("contacts[")]
        assert not [k for k in flat if k.startswith("tags[")]


class TestNormalize:
    def test_strips_strings_only(self):
        assert normalize("  London ") == "London"
        assert normalize(36) == 36
        assert normalize(None) is None

    @pytest.mark.parametrize(
        ("want", "got"),
        [("London", "london"), (36, "36"), (None, ""), (4, 4.5)],
    )
    def test_does_not_coerce_a_wrong_answer_into_a_right_one(self, want, got):
        assert normalize(want) != normalize(got)


class TestScore:
    def test_perfect_match(self):
        result = score(_record(), _record())
        assert result.matched == result.total
        assert result.ratio == 1.0
        assert result.wrong == () and result.missing == ()

    def test_one_wrong_field(self):
        result = score(_record(), _record(age=37))
        assert result.wrong == ("age",)
        assert result.matched == result.total - 1

    def test_failure_scores_zero_against_the_full_denominator(self):
        """A cell that raised must not be excluded - exclusion would flatter it."""
        expected = _record()
        result = score(expected, None)
        assert result.matched == 0
        assert result.total == len(flatten(expected))
        assert result.ratio == 0.0
        assert result.missing == tuple(sorted(flatten(expected)))

    def test_denominator_is_expected_not_produced(self):
        """Emitting fewer fields must not raise the score."""
        expected = _record()
        stingy = score(
            expected, PersonRecord(name="Ada Lovelace", age=36, address=expected.address)
        )
        assert stingy.total == len(flatten(expected))
        assert stingy.ratio < 1.0

    def test_spurious_fields_are_reported_but_do_not_change_the_ratio(self):
        expected = _record(tags=["mathematics"])
        produced = _record(tags=["mathematics", "invented"])
        result = score(expected, produced)
        assert result.spurious == ("tags[1]",)
        assert result.ratio == 1.0

    def test_empty_expected_does_not_divide_by_zero(self):
        assert FieldScore(matched=0, total=0).ratio == 1.0


class TestRecallAndInvented:
    """A correct None is a match, so field accuracy pays for extracting nothing; these don't."""

    def test_splits_the_expected_fields_by_gold_value(self):
        expected = {"name": "Ada", "age": 36, "city": "London", "email": None, "phone": None}
        produced = {"name": "Ada", "age": 37, "email": "ada@x.org", "phone": None}
        result = score(expected, produced)
        # name right, age wrong, city missing; phone rightly null, email filled against gold.
        assert (result.matched, result.total) == (2, 5)
        assert (result.recall_matched, result.recall_total) == (1, 3)
        assert result.invented == 1

    def test_extracting_nothing_scores_zero_recall(self):
        expected = {"name": "Ada", "email": None, "phone": None, "fax": None}
        result = score(expected, dict.fromkeys(expected))
        assert result.ratio == 0.75
        assert (result.recall_matched, result.recall_total, result.invented) == (0, 1, 0)

    def test_list_items_score_per_index_and_an_invented_list_counts_once(self):
        result = score(
            {"tags": ["a", "b"], "aliases": None}, {"tags": ["a", "x"], "aliases": ["y", "z"]}
        )
        assert (result.recall_matched, result.recall_total) == (1, 2)
        assert result.missing == ("aliases",)
        assert result.spurious == ("aliases[0]", "aliases[1]")
        assert result.invented == 1

    def test_an_object_where_the_gold_has_none_is_invented(self):
        result = score(_record(employment=None), _record())
        assert "employment" in result.missing
        assert result.invented == 1

    def test_goes_through_an_align_callable(self):
        expected = {"Company": ["A", "B"], "Person": None, "Date": []}
        produced = {"Company": ["B", "C"], "Person": ["X"], "Date": None}
        result = score(*align_entity_lists(expected, produced))
        # Company aligns to [None, "B", "C"]; Person is filled against a null gold; Date's
        # [] and None are both "nothing" and match.
        assert (result.matched, result.total) == (2, 4)
        assert (result.recall_matched, result.recall_total) == (1, 2)
        assert result.invented == 1

    def test_a_raised_cell_scores_zero_against_the_non_null_denominator(self):
        result = score({"name": "Ada", "age": 36, "email": None}, None)
        assert (result.recall_matched, result.recall_total, result.invented) == (0, 2, 0)

    def test_aggregate_ranks_the_empty_reply_last_on_recall(self, tmp_path):
        from benchmarking.dspy_adapters.report import write_accuracy_report

        def row(adapter, produced):
            s = score({"name": "Ada", "email": None, "phone": None, "fax": None}, produced)
            return AccuracyRow(
                adapter=adapter,
                adapter_config="",
                case_id="c",
                outcome=Outcome.OK if produced is not None else Outcome.PARSE_ERROR,
                error_class="",
                wall_s=0.0,
                lm_calls=1,
                response_format_sent="none",
                matched=s.matched,
                total=s.total,
                wrong=s.wrong,
                missing=s.missing,
                spurious=s.spurious,
                total_tokens=None,
                recall_matched=s.recall_matched,
                recall_total=s.recall_total,
                invented=s.invented,
            )

        empty = dict.fromkeys(["name", "email", "phone", "fax"])
        rows = [
            row("nothing", empty),
            row("nothing", empty),
            row("extracts", {**empty, "name": "Ada", "email": "x"}),
            row("extracts", None),
        ]
        nothing, extracts = aggregate_accuracy(rows)
        assert nothing["field_accuracy"] > extracts["field_accuracy"]  # 6/8 vs 3/8
        assert (nothing["recall"], extracts["recall"]) == (0.0, 0.5)
        assert (extracts["invented"], extracts["null_total"]) == (1, 6)

        md_path, _ = write_accuracy_report(rows, tmp_path)
        lines = md_path.read_text().splitlines()
        assert any(line.endswith("| 0.000 (0/2) | 0.000 (0/6) |") for line in lines)
        assert any(line.endswith("| 0.500 (1/2) | 0.167 (1/6) |") for line in lines)


class TestGenerate:
    def test_is_deterministic_for_a_seed(self):
        assert [c.text for c in generate(5, seed=1)] == [c.text for c in generate(5, seed=1)]

    def test_different_seeds_differ(self):
        assert [c.text for c in generate(5, seed=1)] != [c.text for c in generate(5, seed=2)]

    def test_growing_n_extends_rather_than_reshuffles(self):
        """Two runs at different n must stay comparable on their common prefix."""
        small = generate(3, seed=7)
        large = generate(10, seed=7)
        assert [c.case_id for c in small] == [c.case_id for c in large[:3]]
        assert [c.text for c in small] == [c.text for c in large[:3]]

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError, match="non-negative"):
            generate(-1)

    def test_case_ids_are_unique(self):
        cases = generate(40, seed=3)
        assert len({c.case_id for c in cases}) == 40

    def test_cases_offset_scores_a_held_out_slice_and_records_it(self, monkeypatch):
        from benchmarking.dspy_adapters import cli, config, runner

        seen: dict = {}
        monkeypatch.setattr(config, "load_config", lambda: None)
        monkeypatch.setattr(config, "build_lm", lambda cfg: None)
        monkeypatch.setattr(
            runner, "run_accuracy_arm", lambda lm, cases, **kw: seen.setdefault("loaded", cases)
        )
        monkeypatch.setattr(
            cli, "_finish_live_arm", lambda *a, **provenance: seen.update(provenance) or 0
        )
        assert cli.main(["--accuracy", "--cases", "3", "--cases-offset", "2"]) == 0
        assert [c.case_id for c in seen["loaded"]] == ["case-0002", "case-0003", "case-0004"]
        assert (seen["cases"], seen["cases_offset"]) == (3, 2)

    def test_covers_both_branches_of_every_optional(self):
        """If the corpus never omits an optional, it never tests the hard half of the schema."""
        cases = generate(60, seed=11)
        employments = {c.expected.employment is None for c in cases}
        postcodes = {c.expected.address.postcode is None for c in cases}
        empty_contacts = {not c.expected.contacts for c in cases}
        assert employments == {True, False}
        assert postcodes == {True, False}
        assert empty_contacts == {True, False}


class TestRenderIsLossless:
    """The arm is only fair if the prose states every field it grades."""

    @pytest.mark.parametrize("index", range(30))
    def test_every_scored_value_appears_in_the_text(self, index):
        case = generate(30, seed=5)[index]
        for path, value in flatten(case.expected).items():
            if value is None or value == []:
                continue  # absence is scored as None; there is nothing to state
            assert str(value) in case.text, f"{case.case_id}: {path}={value!r} not in prose"

    def test_omitted_optionals_are_absent_from_the_text(self):
        rng = random.Random(0)
        record = _record(employment=None, address=Address(street="12 Baker St", city="London"))
        text = render(record, rng)
        assert "years so far" not in text
        assert "postcode" not in text.lower()


class TestRunAccuracyArm:
    """The arm end-to-end against an adapter-seeded fake — no network, no dspy.LM.

    Δ2 construction rule applies here as it does to the outcomes arm: the fake renders its
    canned answer *through the adapter under test*, so it must be built per cell.
    """

    @staticmethod
    def _perfect_arm(adapter_ids, cases):
        """Run the arm with a fake that answers each case with that case's ground truth.

        The factory is called once per (adapter, case) in the arm's own loop order, so the
        answers are cycled rather than listed once - a single shared list would feed every
        cell case 0's record and score every later case as a total miss.
        """
        import itertools

        from dspy.utils.dummies import DummyLM

        from benchmarking.dspy_adapters.runner import run_accuracy_arm

        answers = itertools.cycle([{"record": c.expected.model_dump()} for c in cases])
        return run_accuracy_arm(
            lambda adapter: DummyLM([next(answers)], adapter=adapter),
            cases,
            adapter_ids=adapter_ids,
            disable_cache=False,  # never mutate global dspy cache state from a test
        )

    def test_perfect_answers_score_one(self):
        cases = generate(2, seed=4)
        rows = self._perfect_arm(["sola-json-sections"], cases)
        assert len(rows) == 2
        assert all(row.ratio == 1.0 and row.exact for row in rows), [
            (r.case_id, r.outcome, r.wrong, r.missing) for r in rows
        ]

    def test_replies_are_recorded_and_replay_reproduces_the_rows(self, tmp_path):
        """The CSV keeps each raw reply, and re-parsing it offline gives the live row back."""
        import csv
        import json

        from benchmarking.dspy_adapters.report import write_accuracy_report
        from benchmarking.dspy_adapters.runner import replay_accuracy

        cases = generate(2, seed=4)
        rows = self._perfect_arm(["sola-json-sections", "sola-yaml-rescue"], cases)
        _, csv_path = write_accuracy_report(rows, tmp_path)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            recorded = list(csv.DictReader(handle))
        assert [json.loads(rec["replies"]) for rec in recorded] == [list(r.replies) for r in rows]
        assert all(len(r.replies) == 1 for r in rows)
        replayed = replay_accuracy(recorded, cases)
        keys = [(r.adapter, r.case_id, r.outcome, r.matched, r.replies) for r in rows]
        assert [(r.adapter, r.case_id, r.outcome, r.matched, r.replies) for r in replayed] == keys

    def test_a_failed_cell_scores_zero_against_its_full_denominator(self):
        """The whole point of routing failures through `score` rather than dropping them."""
        from dspy.utils.dummies import DummyLM

        from benchmarking.dspy_adapters.runner import run_accuracy_arm

        cases = generate(1, seed=4)

        class Boom(DummyLM):
            def forward(self, *a, **kw):
                raise ValueError("no response")

        rows = run_accuracy_arm(
            lambda adapter: Boom([], adapter=adapter),
            cases,
            adapter_ids=["sola-json-sections"],
            disable_cache=False,
        )
        assert rows[0].matched == 0
        assert rows[0].total == len(flatten(cases[0].expected))
        assert rows[0].ratio == 0.0
        non_null = [v for v in flatten(cases[0].expected).values() if v is not None]
        assert (rows[0].recall_total, rows[0].recall, rows[0].invented) == (len(non_null), 0.0, 0)

    def test_report_writes_both_files_and_aggregates_micro(self, tmp_path):
        import csv

        from benchmarking.dspy_adapters.report import write_accuracy_report

        cases = generate(3, seed=6)
        rows = self._perfect_arm(["sola-json-sections", "json"], cases)
        records = aggregate_accuracy(rows)
        assert [r["adapter"] for r in records] == ["sola-json-sections", "json"]
        assert all(r["cases"] == 3 for r in records)
        # Micro, not the mean of per-case ratios: pooled numerator over pooled denominator.
        for record in records:
            group = [row for row in rows if row.adapter == record["adapter"]]
            assert record["field_accuracy"] == sum(g.matched for g in group) / sum(
                g.total for g in group
            )
            assert (record["recall"], record["invented"]) == (1.0, 0)

        md_path, csv_path = write_accuracy_report(rows, tmp_path)
        assert md_path.exists() and csv_path.exists()
        text = md_path.read_text()
        assert "## Extraction accuracy — aggregate" in text
        assert "sola-json-sections" in text
        assert "| recall (non-null gold) | invented (null gold) |" in text
        # The two live arms must never share a file stem.
        assert md_path.name.startswith("accuracy-")
        assert csv_path.read_text().splitlines()[0].startswith("adapter,adapter_config,case_id")
        first = next(csv.DictReader(csv_path.read_text().splitlines()))
        assert (first["recall"], first["invented"]) == ("1.0", "0")
        assert first["recall_matched"] == first["recall_total"] != "0"


@pytest.mark.parametrize(
    ("plain_id", "rescue_id"),
    [
        ("sola-json-sections", "sola-json-rescue"),
        ("sola-jsonish-sections", "sola-jsonish-rescue"),
        ("sola-yaml-sections", "sola-yaml-rescue"),
    ],
)
def test_rescue_cell_differs_from_its_twin_only_at_parse_time(plain_id, rescue_id):
    """A rescue cell must isolate parse-time repair, nothing else.

    If its prompt ever diverges from its twin, the pair stops being a controlled
    comparison and any accuracy delta between them becomes uninterpretable.
    """
    from benchmarking.dspy_adapters.adapters import ADAPTERS
    from benchmarking.dspy_adapters.signatures import SIGNATURE_IDS, SIGNATURES

    plain = ADAPTERS[plain_id].factory()
    rescue = ADAPTERS[rescue_id].factory()
    assert plain.parse_config is None
    assert rescue.parse_config is not None

    for sig_id in SIGNATURE_IDS:
        cell = SIGNATURES[sig_id]
        assert plain.format(cell.signature, [], cell.inputs) == rescue.format(
            cell.signature, [], cell.inputs
        ), sig_id
