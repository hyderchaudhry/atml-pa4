"""The fixed study design. Adaptive selection runs before held-out scoring."""

from dataclasses import dataclass

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
INPUT_SITE = dict(
    zip(PROJECTIONS, ("q_proj", "q_proj", "q_proj", "o_proj", "gate_proj", "gate_proj", "down_proj"))
)
BITS = (2, 3, 4, 8)
ALPHAS = (0.0, 0.5, 1.0)
STRENGTHS = (0.25, 0.5)
GROUP = 128
# Same positions from each of the 128 full calibration windows for exact local output errors.
POSITIONS = (31, 159, 287, 415)
REPETITIONS = 5
WARMUPS = 2
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class Condition:
    family: str
    target: int = 0

    @property
    def name(self):
        return self.family if not self.target else f"{self.family}_{self.target}"

    @property
    def full_hellaswag(self):
        return (
            self.family == "fp16"
            or self.family in ("p1", "p2", "p3")
            and self.target == 50
            or self.family.startswith("q")
            and self.target in (3, 4)
        )


CONDITIONS = [Condition("fp16")]
for family, targets in [
    ("p1", (25, 50, 75)),
    ("p2", (25, 50, 75)),
    ("p3", (50, 75)),
    ("q1", (8, 4, 3)),
    ("q2", (8, 4, 3)),
    ("q3", (4, 3)),
]:
    CONDITIONS.extend(Condition(family, t) for t in targets)
CONDITIONS.append(Condition("wanda_2of4", 50))


def projections(block):
    return {
        name: getattr(block.self_attn if name in PROJECTIONS[:4] else block.mlp, name)
        for name in PROJECTIONS
    }
