"""Tests for JevAdapter / JevLM: signature -> Jev decision request, answers -> typed outputs."""

from __future__ import annotations

import asyncio
import enum
import io
import json
import math
from typing import Annotated, Any, Literal
from unittest import mock

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
import pydantic  # noqa: E402
from dspy.clients.cache import Cache  # noqa: E402

from llm_schema_lite.dspy_integration import JevAdapter, JevLM, SemIfLM  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap DSPy's global (disk-backed) cache for an empty in-memory one per test."""
    monkeypatch.setattr(
        dspy, "cache", Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=None)
    )


class Team(enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


class Route(pydantic.BaseModel):
    team: Team = pydantic.Field(description="Which team should handle this?")
    escalate: bool


class Triage(dspy.Signature):
    """Triage a support message."""

    message: str = dspy.InputField()
    is_urgent: Annotated[
        bool,
        pydantic.Field(
            json_schema_extra={
                "jev": {
                    "threshold": 0.8,
                    "criteria": {"true": "Time-sensitive", "false": "Not urgent"},
                }
            }
        ),
    ] = dspy.OutputField(desc="Does this message convey urgency?")
    frustration: Annotated[
        Literal["calm", "frustrated", "angry"],
        pydantic.Field(json_schema_extra={"jev": {"type": "score"}}),
    ] = dspy.OutputField()
    route: Route = dspy.OutputField()


ANSWERS: dict[str, Any] = {
    "is_urgent": {"type": "noul", "noul": 0.7},
    "frustration": {
        "type": "score",
        "score": 2.1,
        "confidence": 0.6,
        "probabilities": {"1": 0.1, "2": 0.7, "3": 0.2},
    },
    "route.team": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.9,
        "probabilities": {"billing": 0.95, "technical": 0.05},
    },
    "route.escalate": {"type": "noul", "noul": 0.6},
}


class _StubJev(dspy.BaseLM):  # type: ignore[misc]
    """Returns canned Jev answers and records the payload it was sent."""

    def forward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **_: Any
    ) -> Any:
        import litellm

        self.payload = json.loads((messages or [])[-1]["content"])
        message = {"role": "assistant", "content": json.dumps(ANSWERS)}
        return litellm.ModelResponse(model=self.model, choices=[{"message": message}])


def test_format_builds_state_and_typed_questions() -> None:
    [message] = JevAdapter().format(Triage, demos=[], inputs={"message": "Payouts failing!"})
    payload = json.loads(message["content"])

    assert payload["state"] == {"message": "Payouts failing!"}
    qs = payload["questions"]
    assert set(qs) == {"is_urgent", "frustration", "route.team", "route.escalate"}
    assert qs["is_urgent"] == {
        "type": "noul",
        "instructions": {
            "task": "Triage a support message.",
            "question": "Does this message convey urgency?",
        },
        "criteria": {"true": "Time-sensitive", "false": "Not urgent"},
    }
    assert qs["frustration"]["type"] == "score"
    assert qs["frustration"]["criteria"] == ["calm", "frustrated", "angry"]
    assert qs["route.team"]["type"] == "choice"
    assert qs["route.team"]["criteria"] == {"billing": "billing", "technical": "technical"}
    assert qs["route.team"]["instructions"]["question"] == "Which team should handle this?"
    assert qs["route.escalate"]["instructions"]["question"] == "route.escalate"


def test_unsupported_output_type_raises() -> None:
    class Extract(dspy.Signature):
        text: str = dspy.InputField()
        name: str = dspy.OutputField()

    with pytest.raises(TypeError, match="name"):
        JevAdapter().format(Extract, demos=[], inputs={"text": "hi"})


def test_predict_end_to_end_decodes_answers() -> None:
    lm = _StubJev(model="typesafe/jev-1.13")
    with dspy.context(lm=lm, adapter=JevAdapter()):
        pred = dspy.Predict(Triage)(message="Payouts failing!")

    assert set(lm.payload) == {"state", "questions"}  # JevLM owns the model name
    assert pred.is_urgent is False  # 0.7 < per-field threshold 0.8
    assert pred.frustration == "frustrated"  # argmax level, base-agnostic
    assert pred.route == Route(team=Team.BILLING, escalate=True)  # default threshold 0.5
    assert pred.jev["route.team"]["confidence"] == 0.9


def test_jev_lm_posts_payload_and_returns_answers() -> None:
    body = json.dumps(
        {"model": "typesafe/jev-1.13", "answers": ANSWERS, "usage": {"input_tokens": 12}}
    )
    lm = JevLM(api_key="k")
    payload = {"state": "s", "questions": {}}
    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body.encode())) as urlopen:
        out = lm(messages=[{"role": "user", "content": json.dumps(payload)}])

    req = urlopen.call_args.args[0]
    assert req.full_url == "https://openrouter.ai/api/alpha/decisions"
    assert req.get_header("Authorization") == "Bearer k"
    assert json.loads(req.data) == {"model": "typesafe/jev-1.13", **payload}
    assert json.loads(out[0]) == ANSWERS


def _jev_body() -> io.BytesIO:
    body = {"model": "typesafe/jev-1.13", "answers": ANSWERS, "usage": {"input_tokens": 12}}
    return io.BytesIO(json.dumps(body).encode())


def test_predict_acall_uses_jev_lm_async() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=lambda *_: _jev_body()):
        with dspy.context(lm=JevLM(api_key="k"), adapter=JevAdapter()):
            pred = asyncio.run(dspy.Predict(Triage).acall(message="Payouts failing!"))

    assert pred.frustration == "frustrated"
    assert pred.jev["route.team"]["choice"] == "billing"


def test_jev_lm_state_round_trips_without_api_key() -> None:
    state = JevLM(api_key="secret", url="https://api.typesafe.ai/v1/systemone").dump_state()

    assert "secret" not in json.dumps(state)
    lm = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(lm, JevLM)
    assert lm.url == "https://api.typesafe.ai/v1/systemone"


def test_jev_lm_caches_identical_requests() -> None:
    lm = JevLM(api_key="k")
    msg = [{"role": "user", "content": json.dumps({"state": "s", "questions": {}})}]
    with mock.patch("urllib.request.urlopen", side_effect=lambda *_: _jev_body()) as urlopen:
        first, second = lm(messages=msg), lm(messages=msg)
        assert urlopen.call_count == 1
        assert first == second
        assert lm.history[-1]["response"].cache_hit

        lm(messages=msg, cache=False)
        assert urlopen.call_count == 2


# Letter logprobs the fake local server returns, keyed by the criterion's last line.
LETTER_PROBS = {
    "Does this message convey urgency?": {"A": 0.6, "B": 0.3, "Sure": 0.1},
    "frustration": {"A": 0.1, "B": 0.2, " C": 0.5, "C": 0.2},  # " C" and "C" both count
    "Which team should handle this?": {"B": 0.8, "A": 0.2},
    "route.escalate": {"A": 0.9, "B": 0.1},
    "no letter": {"Yes": 1.0},
}


SENT: list[dict[str, Any]] = []


def _fake_completion(request: dict[str, Any], **_: Any) -> Any:
    """Stands in for ``litellm_completion`` (a real function: DSPy's cache reads its name)."""
    import litellm

    SENT.append(request)
    assert request["max_tokens"] == 1 and request["logprobs"] and request["top_logprobs"] == 20
    content = request["messages"][-1]["content"]
    if not isinstance(content, str):
        # A multimodal turn splits the payload where the image sat, between two JSON
        # string quotes, so dropping the blocks leaves the surrounding JSON parseable.
        content = "".join(block["text"] for block in content if block["type"] == "text")
    criterion = json.loads(content)["criterion"]
    top = [
        {"token": t, "logprob": math.log(p), "bytes": None}
        for t, p in LETTER_PROBS[criterion.splitlines()[-1]].items()
    ]
    logprobs = {"content": [{"token": top[0]["token"], "logprob": 0.0, "top_logprobs": top}]}
    return litellm.ModelResponse(
        model="local",
        choices=[{"message": {"role": "assistant", "content": "A"}, "logprobs": logprobs}],
    )


async def _afake_completion(request: dict[str, Any], **kwargs: Any) -> Any:
    return _fake_completion(request, **kwargs)


def test_semif_lm_reads_option_letter_logprobs() -> None:
    lm = SemIfLM("openai/local", api_base="http://localhost:8000/v1")
    SENT.clear()
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=lm, adapter=JevAdapter()):
            pred = dspy.Predict(Triage)(message="Payouts failing!")

    assert len(SENT) == 4  # one single-token request per question
    sent = json.loads(SENT[0]["messages"][-1]["content"])
    assert sent["evidence"] == {"message": "Payouts failing!"}
    assert sent["options"][0] == {"letter": "A", "description": "Time-sensitive"}

    assert pred.jev["is_urgent"]["noul"] == pytest.approx(0.6 / 0.9)  # "Sure" is not a letter
    assert pred.is_urgent is False  # 0.67 < per-field threshold 0.8
    assert pred.frustration == "angry"  # C: 0.5 + 0.2
    assert pred.jev["frustration"]["probabilities"] == pytest.approx({"0": 0.1, "1": 0.2, "2": 0.7})
    assert pred.route == Route(team=Team.TECHNICAL, escalate=True)


def test_semif_lm_async_matches_sync() -> None:
    lm = SemIfLM("openai/local")
    with mock.patch("dspy.clients.lm.alitellm_completion", new=_afake_completion):
        with dspy.context(lm=lm, adapter=JevAdapter()):
            pred = asyncio.run(dspy.Predict(Triage).acall(message="Payouts failing!"))

    assert pred.frustration == "angry"
    assert pred.route.team is Team.TECHNICAL


def test_semif_lm_raises_when_no_letter_is_in_top_logprobs() -> None:
    lm = SemIfLM("openai/local")
    payload = {
        "state": "s",
        "questions": {"q": {"type": "noul", "instructions": {"task": "", "question": "no letter"}}},
    }
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with pytest.raises(ValueError, match="No option letter"):
            lm(messages=[{"role": "user", "content": json.dumps(payload)}])


PNG = (  # smallest valid PNG: one transparent pixel
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAA"
    "CklEQVR4nGP4DwABAQEAWk1v8QAAAABJRU5ErkJggg=="
)


class Screenshots(dspy.Signature):
    """Read the screenshots."""

    before: dspy.Image = dspy.InputField()
    after: dspy.Image = dspy.InputField()
    is_urgent: bool = dspy.OutputField(desc="Does this message convey urgency?")


def test_semif_lm_sends_image_inputs_as_content_blocks() -> None:
    """dspy.Image inputs reach the server as image blocks, not as base64 in the payload text."""
    SENT.clear()
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=SemIfLM("openai/local"), adapter=JevAdapter()):
            pred = dspy.Predict(Screenshots)(before=dspy.Image(PNG), after=dspy.Image(PNG))

    [sent] = SENT
    content = sent["messages"][-1]["content"]
    assert [b for b in content if b["type"] == "image_url"] == [
        {"type": "image_url", "image_url": {"url": PNG}},
        {"type": "image_url", "image_url": {"url": PNG}},
    ]
    [text] = [b["text"] for b in content if b["type"] == "text"]
    assert PNG not in text
    # Identical images still get distinct placeholders, and the payload stays valid JSON.
    assert json.loads(text)["evidence"] == {"before": "<<image 1>>", "after": "<<image 2>>"}
    assert pred.is_urgent is True  # 0.6 >= the default 0.5 threshold


def test_semif_lm_leaves_a_text_only_payload_as_a_plain_string() -> None:
    """No custom type, no content blocks: SemIf's prompt stays byte-for-byte what it was."""
    SENT.clear()
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=SemIfLM("openai/local"), adapter=JevAdapter()):
            dspy.Predict(Triage)(message="Payouts failing!")

    assert all(isinstance(s["messages"][-1]["content"], str) for s in SENT)


def test_semif_lm_state_round_trips() -> None:
    state = SemIfLM(
        "openai/local",
        top_logprobs=5,
        question_first=True,
        parallel_questions=True,
        calibration_temperature=2.0,
        api_base="http://h/v1",
    ).dump_state()
    lm = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(lm, SemIfLM)
    assert lm.top_logprobs == 5
    assert lm.question_first is True
    assert lm.parallel_questions is True
    assert lm.calibration_temperature == 2.0


@pytest.mark.parametrize(
    ("extra_body", "sent"),
    [
        (None, {"chat_template_kwargs": {"enable_thinking": False}}),
        ({"top_k": 1}, {"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}}),
        (  # an explicit choice wins
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
    ],
)
def test_semif_lm_turns_thinking_off_unless_told_otherwise(
    extra_body: dict[str, Any] | None, sent: dict[str, Any]
) -> None:
    """A thinking model would open with <think>, not a letter; SemIf renders with it off."""
    SENT.clear()
    kwargs = {"extra_body": extra_body} if extra_body else {}
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=SemIfLM("openai/local", **kwargs), adapter=JevAdapter()):
            dspy.Predict(Triage)(message="Payouts failing!")

    assert all(s["extra_body"] == sent for s in SENT)


@pytest.mark.parametrize(("parallel", "most_in_flight"), [(False, 1), (True, 4)])
def test_semif_lm_async_asks_one_states_questions_in_turn(
    parallel: bool, most_in_flight: int
) -> None:
    """In turn, llama.cpp reads the shared state once; in parallel, once per server slot."""
    in_flight, seen = 0, []

    async def fake(request: dict[str, Any], **kwargs: Any) -> Any:
        nonlocal in_flight
        in_flight += 1
        seen.append(in_flight)
        await asyncio.sleep(0.01)  # long enough for DSPy 3.4's awaits to interleave
        in_flight -= 1
        return _fake_completion(request, **kwargs)

    lm = SemIfLM("openai/local", parallel_questions=parallel)
    with mock.patch("dspy.clients.lm.alitellm_completion", new=fake):
        with dspy.context(lm=lm, adapter=JevAdapter()):
            pred = asyncio.run(dspy.Predict(Triage).acall(message="Payouts failing!"))

    assert max(seen) == most_in_flight
    assert pred.frustration == "angry"


def test_semif_lm_calibration_temperature_softens_the_probabilities() -> None:
    """p_i ** (1 / T), renormalised: the same as softmax(logits / T) over the options."""
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=SemIfLM("openai/local", calibration_temperature=2.0)):
            with dspy.context(adapter=JevAdapter()):
                pred = dspy.Predict(Triage)(message="Payouts failing!")

    # A: 0.6, B: 0.3 -> sqrt(2/3) / (sqrt(2/3) + sqrt(1/3))
    assert pred.jev["is_urgent"]["noul"] == pytest.approx(2**0.5 / (2**0.5 + 1))
    assert pred.frustration == "angry"  # the argmax never moves


@pytest.mark.parametrize(
    ("question_first", "order"),
    [(False, ["evidence", "criterion", "options"]), (True, ["criterion", "options", "evidence"])],
)
def test_semif_lm_question_first_moves_the_evidence_last(
    question_first: bool, order: list[str]
) -> None:
    """Evidence last lets fanned-out requests share the question in the server's KV cache."""
    SENT.clear()
    with mock.patch("dspy.clients.lm.litellm_completion", new=_fake_completion):
        with dspy.context(lm=SemIfLM("openai/local", question_first=question_first)):
            with dspy.context(adapter=JevAdapter()):
                pred = dspy.Predict(Triage)(message="Payouts failing!")

    assert all(list(json.loads(s["messages"][-1]["content"])) == order for s in SENT)
    assert pred.frustration == "angry"  # same readout either way


class _LazyLogprobs(pydantic.BaseModel):
    """Stands in for a litellm response unpickled from DSPy's disk cache in a new process:
    the forward reference leaves its serializer unbuilt (``MockValSer``) until first dump."""

    content: list[_Token]


class _Token(pydantic.BaseModel):
    token: str
    logprob: float
    top_logprobs: list[dict[str, Any]]


def test_semif_lm_reads_logprobs_whose_serializer_is_not_built_yet() -> None:
    from types import SimpleNamespace

    from llm_schema_lite.dspy_integration.adapters.semif_lm import _answer

    top = [{"token": "B", "logprob": math.log(0.75)}, {"token": "A", "logprob": math.log(0.25)}]
    lazy = _LazyLogprobs.model_construct(content=[_Token(token="B", logprob=0.0, top_logprobs=top)])
    assert type(_LazyLogprobs.__pydantic_serializer__).__name__ == "MockValSer"
    response = SimpleNamespace(choices=[SimpleNamespace(logprobs=lazy)])

    answer = _answer({"type": "noul"}, {"true": "Yes", "false": "No"}, response)

    assert answer["noul"] == pytest.approx(0.25)


def _fake_pairs(request: dict[str, Any], **_: Any) -> Any:
    """First read: whole-label tokens plus a lone prefix "A"; after "A", the second letter."""
    import litellm

    SENT.append(request)
    last = request["messages"][-1]
    probs = (
        {"C": 0.75, "D": 0.25, " apple": 0.5}  # junk is renormalised away
        if last["role"] == "assistant"
        else {"A": 0.6, "BC": 0.3, "Sure": 0.1, "B": 1e-4}  # "B" is under the 1% floor
    )
    top = [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]
    logprobs = {"content": [{"token": top[0]["token"], "logprob": 0.0, "top_logprobs": top}]}
    return litellm.ModelResponse(
        model="local",
        choices=[{"message": {"role": "assistant", "content": "A"}, "logprobs": logprobs}],
    )


async def _afake_pairs(request: dict[str, Any], **kwargs: Any) -> Any:
    return _fake_pairs(request, **kwargs)


def test_semif_lm_reads_more_than_16_options_through_two_letter_labels() -> None:
    """17-256 options get labels AA..PP: P(label) = P(whole token) + P(first) * P(second)."""
    criteria = {f"o{i}": f"option {i}" for i in range(20)}  # AA..AP, then BA..BD
    payload = {
        "state": "s",
        "questions": {
            "q": {
                "type": "choice",
                "criteria": criteria,
                "instructions": {"task": "", "question": ""},
            }
        },
    }
    messages = [{"role": "user", "content": json.dumps(payload)}]
    for patch, call in [
        ("litellm_completion", lambda lm: lm(messages=messages)),
        ("alitellm_completion", lambda lm: asyncio.run(lm.acall(messages=messages))),
    ]:
        SENT.clear()
        fake = _fake_pairs if patch == "litellm_completion" else _afake_pairs
        with mock.patch(f"dspy.clients.lm.{patch}", new=fake):
            [out] = call(SemIfLM("openai/local"))
        answer = json.loads(out)["q"]

        sent = json.loads(SENT[0]["messages"][-1]["content"])
        assert [o["letter"] for o in sent["options"]][15:] == ["AP", "BA", "BB", "BC", "BD"]
        assert len(SENT) == 2  # one follow-up, for prefix "A" only
        assert SENT[1]["messages"][-1] == {"role": "assistant", "content": "A"}
        # AC = 0.6 * 0.75, AD = 0.6 * 0.25, BC = 0.3 whole; renormalised over 0.9
        assert answer["probabilities"]["o2"] == pytest.approx(0.5)
        assert answer["probabilities"]["o3"] == pytest.approx(1 / 6)
        assert answer["probabilities"]["o18"] == pytest.approx(1 / 3)
        assert answer["choice"] == "o2"


def test_semif_lm_rejects_more_than_256_options() -> None:
    payload = {
        "state": "s",
        "questions": {
            "q": {
                "type": "choice",
                "criteria": {str(i): "x" for i in range(257)},
                "instructions": {"task": "", "question": ""},
            }
        },
    }
    with pytest.raises(ValueError, match="2-256 options"):
        SemIfLM("openai/local")(messages=[{"role": "user", "content": json.dumps(payload)}])
