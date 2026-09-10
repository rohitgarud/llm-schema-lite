"""Tests for the third-party accuracy corpora: rows -> cases, and NER list alignment.

No network. Rows are built by hand in the shape the pinned Hugging Face revisions have - a
text column plus a Python-literal ``ground_truth`` string - with invented values, because
the upstream datasets state no licence and are never copied into this repo.
"""

from __future__ import annotations

from benchmarking.dspy_adapters.accuracy import null_floor, score
from benchmarking.dspy_adapters.external import CORPORA, align_entity_lists, cases_from_rows

CLAIM = {
    "header": {
        "claim_id": "CLM-000001",
        "report_date": "2024-01-02",
        "incident_date": "2024-01-01",
        "reported_by": "Test Person",
        "channel": "Email",
    },
    "policy_details": None,
    "insured_objects": [
        {
            "object_id": "PROP-000001",
            "object_type": "Building",
            "make_model": "Single Family Home",
            "year": 1970,
            "location_address": "1 Test Street",
            "estimated_value": None,
        }
    ],
    "incident_description": {
        "incident_type": "vandalism",
        "location_type": "residence_exterior",
        "estimated_damage_amount": 1000,
        "police_report_number": None,
    },
}


def test_rows_become_cases_bound_to_the_corpus_signature():
    rows = [{"claim_text": "a claim", "ground_truth": repr(CLAIM)}]
    [case] = cases_from_rows("insurance-claims", rows)
    assert case.case_id == "insurance-claims-0000"
    assert case.expected == CLAIM
    assert case.signature is CORPORA["insurance-claims"].signature
    assert (case.input_field, case.output_field) == ("claim_text", "claim")


class TestAlignEntityLists:
    def test_order_duplicates_and_empty_vs_none_do_not_matter(self):
        expected = {"Company": ["A", "B"], "Date": None}
        produced = {"Company": ["B", "A", "A"], "Date": []}
        result = score(*align_entity_lists(expected, produced))
        assert (result.matched, result.total, result.spurious) == (3, 3, ())

    def test_a_missed_entity_is_wrong_and_an_invented_one_is_spurious(self):
        result = score(*align_entity_lists({"Company": ["A", "B"]}, {"Company": ["B", "C"]}))
        assert (result.matched, result.total) == (1, 2)
        assert result.wrong == ("Company[0]",)
        assert result.spurious == ("Company[2]",)

    def test_entities_for_an_empty_category_cost_its_field(self):
        result = score(*align_entity_lists({"Person": None}, {"Person": ["X"]}))
        assert (result.matched, result.total) == (0, 1)

    def test_a_failed_cell_keeps_the_aligned_denominator(self):
        expected, produced = align_entity_lists({"Company": ["A", "A"], "Date": []}, None)
        assert produced is None
        assert score(expected, None).total == 2


def test_perfect_answer_on_a_nested_dated_corpus_scores_one(tmp_path):
    """Routing (signature, input, output field) and JSON-mode dates, end to end."""
    from dspy.utils.dummies import DummyLM

    from benchmarking.dspy_adapters.report import write_accuracy_report
    from benchmarking.dspy_adapters.runner import run_accuracy_arm

    cases = cases_from_rows("insurance-claims", [{"claim_text": "x", "ground_truth": repr(CLAIM)}])
    rows = run_accuracy_arm(
        lambda adapter: DummyLM([{"claim": CLAIM}], adapter=adapter),
        cases,
        adapter_ids=["sola-json-sections"],
        disable_cache=False,  # never mutate global dspy cache state from a test
    )
    assert rows[0].exact, (rows[0].outcome, rows[0].wrong, rows[0].missing)

    md_path, _ = write_accuracy_report(
        rows, tmp_path, corpus="insurance-claims", null_floor=null_floor(cases)
    )
    assert md_path.name.startswith("accuracy-insurance-claims-")
    text = md_path.read_text()
    assert "third-party `insurance-claims` corpus" in text
    assert "rendered from" not in text  # the synthetic corpus's note must not leak in
    # policy_details is CLAIM's only top-level None: 1 of its 16 leaves is right by saying
    # nothing (a None *inside* a nested object is missed, since the reply has no parent).
    assert f"**All-null floor: {1 / 16:.3f}**" in text


def test_patient_notes_gold_and_a_lenient_reply_meet_in_one_canonical_form():
    """Upstream's leniencies: case, state abbreviations, written-out birth dates, empty ==
    null, a one-item address line list == its string, and unrequested gold keys dropped."""
    gold = {
        "record_id": 7,
        "name": {"family": "Doe", "given": ["Jane"], "prefix": "Ms."},
        "age": None,
        "gender": "female",
        "birthDate": "1988-11-12",
        "phone": None,
        "email": None,
        "maritalStatus": "Married",
        "primaryLanguage": "Spanish",
        "address": {
            "line": "1 Test St",
            "city": "Boston",
            "state": "Massachusetts",
            "postalCode": "02111",
            "country": "US",
        },
        "practitioner": [
            {
                "name": None,
                "phone": None,
                "email": None,
                "address": {
                    "line": ["9 Clinic Rd"],
                    "city": "Boston",
                    "state": None,
                    "postalCode": None,
                    "country": "US",
                },
                "encounter_period": {"start": "2020-01-01"},
            }
        ],
        "allergy": {"substance": None},
        "immunization": [],
    }
    reply = {
        "name": {"family": "DOE", "given": ["jane"], "prefix": "Ms."},
        "age": None,
        "gender": "female",
        "birthDate": "November 12, 1988",
        "phone": None,
        "email": None,
        "maritalStatus": "Married",
        "primaryLanguage": "spanish",
        "address": {
            "line": "1 Test St",
            "city": "Boston",
            "state": "MA",
            "postalCode": "02111",
            "country": "US",
        },
        "practitioner": [
            {
                "name": None,
                "phone": None,
                "email": None,
                "address": {
                    "line": "9 clinic rd",
                    "city": "Boston",
                    "state": None,
                    "postalCode": None,
                    "country": "US",
                },
            }
        ],
        "allergy": None,
        "immunization": None,
    }
    [case] = cases_from_rows("patient-notes", [{"note": "x", "ground_truth": gold}])
    assert case.input_field == "note" and case.output_field == "patient_record"
    result = score(*case.align(case.expected, reply))
    assert (result.matched, result.wrong, result.missing, result.spurious) == (
        result.total,
        (),
        (),
        (),
    )


def test_null_floor_goes_through_align():
    """An all-None NER reply matches exactly the categories whose gold is empty."""
    gold = {"Company": ["A", "B"], "Date": None, "Person": []}
    cases = cases_from_rows("financial-ner", [{"text": "x", "ground_truth": repr(gold)}])
    assert null_floor(cases) == 2 / 4
