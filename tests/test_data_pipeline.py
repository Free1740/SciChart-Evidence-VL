"""Unit tests for the data pipeline: src/data/teacher_review.py,
src/data/answer_check.py and src/data/difficulty.py.

These three modules are the pipeline added in commit de06303f
(scripts/build_dataset.py runs review -> verify -> difficulty filter)
and had no tests of their own. No torch dependency.
Run with: pytest tests/test_data_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.answer_check import verify_answer, verify_answers  # noqa: E402
from src.data.difficulty import filter_hard, score_difficulty  # noqa: E402
from src.data.schema import ChartQASample, EvidenceRegion  # noqa: E402
from src.data.teacher_review import (  # noqa: E402
    review_sample,
    review_samples,
)

CODE = """import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
plt.plot(x, [0.80, 0.85, 0.90, 0.93], label="Model-A")
plt.plot(x, [0.70, 0.72, 0.75, 0.78], label="Model-B")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
"""

HARD_CODE = """import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
plt.plot(x, [0.80, 0.85, 0.90, 0.93], label="Model-A", color="blue")
plt.plot(x, [0.70, 0.72, 0.75, 0.78], label="Model-B", color="red")
plt.plot(x, [0.65, 0.70, 0.74, 0.76], label="Model-C", color="green")
plt.plot(x, [0.60, 0.62, 0.65, 0.66], label="Model-D", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy over training epochs")
plt.legend()
plt.grid(True)
"""


def _sample(**overrides):
    fields = {
        "sample_id": "s1",
        "question": "Which model has the highest accuracy at the final epoch?",
        "answer": "Model-A",
        "plotting_code": CODE,
        "evidence": [EvidenceRegion(label="Model-A", code_ref="Model-A")],
    }
    fields.update(overrides)
    return ChartQASample(**fields)


def _hard_sample(sample_id="hard-1"):
    return ChartQASample(
        sample_id=sample_id,
        question=(
            "Across all four training epochs shown in the chart, which model "
            "achieves the highest accuracy at the final epoch, and by how much "
            "does it exceed the second-best model?"
        ),
        answer="Model-A",
        plotting_code=HARD_CODE,
        evidence=[
            EvidenceRegion(label="Model-A", code_ref="Model-A"),
            EvidenceRegion(label="Model-B", code_ref="Model-B"),
            EvidenceRegion(label="Model-C", code_ref="Model-C"),
        ],
    )


# --- Teacher review -------------------------------------------------------


def test_teacher_approves_traceable_sample():
    review = review_sample(_sample())
    assert review.verdict == "approve"
    assert review.score == 1.0
    assert review.reviewer == "teacher"


def test_teacher_approves_answer_traceable_via_question():
    # Answer is not in the plotting code, but is in the question, and the
    # evidence ref ("plt.plot") is in the code, so it is still traceable.
    sample = _sample(
        question="Is Model-A the best model at the final epoch?",
        plotting_code="import matplotlib.pyplot as plt\nplt.plot([1, 2], [0.8, 0.9])\n",
        evidence=[EvidenceRegion(label="accuracy curve", code_ref="plt.plot")],
    )
    assert review_sample(sample).verdict == "approve"


def test_teacher_rejects_empty_question():
    assert review_sample(_sample(question="  ")).verdict == "reject"
    assert review_sample(_sample(question="  ")).score == 0.0


def test_teacher_rejects_empty_answer():
    assert review_sample(_sample(answer="")).verdict == "reject"


def test_teacher_rejects_empty_plotting_code():
    assert review_sample(_sample(plotting_code="")).verdict == "reject"


def test_teacher_rejects_missing_evidence():
    assert review_sample(_sample(evidence=[])).verdict == "reject"


def test_teacher_rejects_evidence_not_in_code():
    sample = _sample(
        evidence=[EvidenceRegion(label="Model-Z", code_ref="Model-Z")]
    )
    review = review_sample(sample)
    assert review.verdict == "reject"
    assert "Model-Z" in review.comments[0]


def test_teacher_needs_revision_when_answer_untraceable():
    # Evidence is in the code, but the answer appears nowhere.
    sample = _sample(answer="Model-Z wins by a lot")
    review = review_sample(sample)
    assert review.verdict == "needs_revision"
    assert review.score == 0.7


def test_teacher_review_samples_preserves_order_and_reviewer():
    samples = [_sample(), _sample(sample_id="s2", question="")]
    reviews = review_samples(samples, reviewer="teacher-v2")
    assert [r.verdict for r in reviews] == ["approve", "reject"]
    assert all(r.reviewer == "teacher-v2" for r in reviews)


# --- Answer verification --------------------------------------------------


def test_verify_answer_in_plotting_code():
    assert verify_answer(_sample()) is True


def test_verify_answer_in_question():
    sample = _sample(
        question="Is Model-A the best model?",
        plotting_code="import matplotlib.pyplot as plt\nplt.plot([1, 2], [0.8, 0.9])\n",
        evidence=[EvidenceRegion(label="accuracy curve", code_ref="plt.plot")],
    )
    assert verify_answer(sample) is True


def test_verify_answer_via_evidence_label():
    # The answer itself is not in the code, but it refers to the key
    # evidence label "Model-A", which is in the plotting code.
    sample = _sample(answer="Model-A wins")
    assert verify_answer(sample) is True


def test_verify_answer_false_when_untraceable():
    # Neither the answer nor any evidence label appears in the code
    # or the question, so there is no trace to the chart at all.
    sample = _sample(
        answer="Model-Z wins by a lot",
        evidence=[EvidenceRegion(label="Model-Z", code_ref="Model-Z")],
    )
    assert verify_answer(sample) is False


def test_verify_answer_false_for_empty_answer():
    assert verify_answer(_sample(answer="   ")) is False


def test_verify_answers_keyed_by_sample_id():
    result = verify_answers([_sample(), _sample(sample_id="s2", answer="")])
    assert result == {"s1": True, "s2": False}


# --- Difficulty scoring / filtering ---------------------------------------


def test_difficulty_score_is_bounded():
    assert 0.0 <= score_difficulty(_sample()) <= 1.0
    assert 0.0 <= score_difficulty(_hard_sample()) <= 1.0


def test_difficulty_hard_sample_scores_higher_than_easy():
    assert score_difficulty(_hard_sample()) > score_difficulty(_sample())
    # The hard fixture is designed to clear the 0.7 default threshold.
    assert score_difficulty(_hard_sample()) >= 0.7
    assert score_difficulty(_sample()) < 0.7


def test_difficulty_score_is_exact_for_known_fixture():
    # Easy fixture: question 56 chars -> 56/120*0.3; 1 evidence -> 1/3*0.3;
    # 6 code lines -> 6/20*0.2; 2 plot calls -> 2/5*0.2.
    expected = (56 / 120.0) * 0.3 + (1 / 3.0) * 0.3 + (6 / 20.0) * 0.2 + (2 / 5.0) * 0.2
    assert abs(score_difficulty(_sample()) - expected) < 1e-9


def test_filter_hard_keeps_only_hard_samples():
    kept = filter_hard([_sample(), _hard_sample()])
    assert [s.sample_id for s in kept] == ["hard-1"]


def test_filter_hard_writes_difficulty_back():
    samples = [_sample(), _hard_sample()]
    filter_hard(samples)
    assert samples[0].difficulty == score_difficulty(samples[0])
    assert samples[1].difficulty == score_difficulty(samples[1])


def test_filter_hard_custom_threshold():
    assert len(filter_hard([_sample(), _hard_sample()], threshold=0.0)) == 2
    assert filter_hard([_sample(), _hard_sample()], threshold=1.0) == []


# --- Pipeline composition: review -> verify -> filter ----------------------


def test_pipeline_keeps_hard_approved_sample_drops_bad_one():
    samples = [_hard_sample(), _sample(sample_id="bad-1", answer="")]
    reviewed = [s for s in samples if review_sample(s).verdict == "approve"]
    verified = [s for s in reviewed if verify_answer(s)]
    kept = filter_hard(verified)
    assert [s.sample_id for s in kept] == ["hard-1"]
