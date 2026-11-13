"""Counterfactual construction: programmatically edit plotting code.

EP (Evidence Preserved): keep the key evidence, apply only answer-irrelevant
perturbations (e.g. change a non-evidence curve colour).
EA (Evidence Ablated): remove the key evidence (a curve, a bar, a metric or
local data) from the plotting code, so the chart no longer supports the answer.

This module works on plotting-code strings only; it does not need any real
data file. Rendering is optional and isolated in ``render_code``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import ChartQASample, CounterfactualPair, EvidenceRegion


@dataclass
class CodeEdit:
    """One edit applied to plotting code."""

    description: str
    original_code: str
    edited_code: str


def _lines_with_ref(code: str, code_ref: str) -> list[int]:
    """Return line indices that mention the evidence ``code_ref``."""
    return [i for i, line in enumerate(code.splitlines()) if code_ref in line]


def make_evidence_preserved(code: str, evidence: EvidenceRegion) -> CodeEdit:
    """Build an EP view: keep the evidence, perturb something irrelevant.

    Example perturbation: change the colour of a curve that is *not* the
    evidence curve. If no such curve is found, the code is returned unchanged
    (a valid EP view — identity perturbation).
    """
    lines = code.splitlines()
    ref_lines = set(_lines_with_ref(code, evidence.code_ref or evidence.label))
    for i, line in enumerate(lines):
        if i in ref_lines:
            continue
        # e.g. plt.plot(x, y, color="blue") -> color="gray"
        if 'color=' in line or "color =" in line:
            edited = re.sub(r'color\s*=\s*["\'][^"\']+["\']', 'color="gray"', line)
            if edited != line:
                new_lines = lines.copy()
                new_lines[i] = edited
                return CodeEdit(
                    description="Changed colour of a non-evidence curve (EP)",
                    original_code=code,
                    edited_code="\n".join(new_lines),
                )
    return CodeEdit(
        description="No irrelevant perturbation applied; evidence unchanged (EP)",
        original_code=code,
        edited_code=code,
    )


def make_evidence_ablated(code: str, evidence: EvidenceRegion) -> CodeEdit:
    """Build an EA view: remove the lines that draw the key evidence."""
    ref = evidence.code_ref or evidence.label
    lines = code.splitlines()
    keep = [line for line in lines if ref not in line]
    if len(keep) == len(lines):
        raise ValueError(
            f"Evidence ref {ref!r} not found in plotting code; "
            "cannot build an EA view for this sample."
        )
    return CodeEdit(
        description=f"Removed evidence {ref!r} from plotting code (EA)",
        original_code=code,
        edited_code="\n".join(keep),
    )


class CounterfactualBuilder:
    """Build EP / EA pairs from a :class:`ChartQASample`."""

    def build_pair(self, sample: ChartQASample) -> CounterfactualPair:
        if not sample.evidence:
            raise ValueError(f"Sample {sample.sample_id} has no evidence region.")
        # The first evidence region is the key evidence for this QA pair.
        evidence = sample.evidence[0]
        ep = make_evidence_preserved(sample.plotting_code, evidence)
        ea = make_evidence_ablated(sample.plotting_code, evidence)
        return CounterfactualPair(
            pair_id=f"{sample.sample_id}-cf",
            sample_id=sample.sample_id,
            question=sample.question,
            answer=sample.answer,
            ep_plotting_code=ep.edited_code,
            ea_plotting_code=ea.edited_code,
            ablation_description=ea.description,
            evidence=sample.evidence,
        )

    def build_pairs(self, samples: list[ChartQASample]) -> list[CounterfactualPair]:
        pairs: list[CounterfactualPair] = []
        for sample in samples:
            try:
                pairs.append(self.build_pair(sample))
            except ValueError:
                # Samples whose evidence cannot be located in code are skipped;
                # they are logged by the caller via the returned count.
                continue
        return pairs


def render_code(code: str, output_path: str) -> str:
    """Render plotting code to an image file (optional, matplotlib only).

    The code is executed in an isolated namespace. Returns ``output_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    namespace: dict = {}
    exec(compile(code, "<plotting_code>", "exec"), namespace)  # noqa: S102
    import matplotlib.pyplot as plt

    plt.savefig(output_path)
    plt.close("all")
    return output_path


# --- Example (no real data) -------------------------------------------------
EXAMPLE_CODE = '''import matplotlib.pyplot as plt
x = [1, 2, 3]
plt.plot(x, [0.8, 0.85, 0.9], label="Model-A", color="blue")
plt.plot(x, [0.7, 0.72, 0.75], label="Model-B", color="red")
plt.legend()
'''

if __name__ == "__main__":
    sample = ChartQASample(
        sample_id="demo-001",
        question="Which model has higher accuracy at x=3?",
        answer="Model-A",
        plotting_code=EXAMPLE_CODE,
        evidence=[EvidenceRegion(label="Model-A", code_ref="Model-A")],
    )
    pair = CounterfactualBuilder().build_pair(sample)
    print("EP keeps Model-A:", "Model-A" in pair.ep_plotting_code)
    print("EA removes Model-A:", "Model-A" not in pair.ea_plotting_code)
    print("Ablation:", pair.ablation_description)
