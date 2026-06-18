"""
afh.scorecard — Aggregate the per-axis faithfulness scores into a scorecard.

Per clip: combine axes 1-3 (and report accuracy/minADE separately, never mixed in).
Per dataset: average across clips, plus the distribution of contradictions / hallucinations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from afh.trace import ClipRecord
from afh.axes.consistency import trace_consistency
from afh.axes.grounding import trace_grounding
from afh.axes.stability import stability


@dataclass
class ClipScore:
    clip_id: str
    consistency: Optional[float] = None    # axis 1, mean over samples
    grounding: Optional[float] = None      # axis 2, mean over samples
    stability: Optional[float] = None      # axis 3
    contradiction: bool = False
    min_ade: Optional[float] = None        # accuracy, reported alongside (not mixed in)
    n_samples: int = 0

    def faithfulness(self) -> Optional[float]:
        """Combined faithfulness = mean of available axes (1-3). Accuracy excluded."""
        parts = [p for p in (self.consistency, self.grounding, self.stability) if p is not None]
        return sum(parts) / len(parts) if parts else None

    def to_dict(self):
        d = asdict(self)
        d["faithfulness"] = self.faithfulness()
        return d


def score_clip(rec: ClipRecord) -> ClipScore:
    """Compute axes 1-3 for one clip record (N samples)."""
    # axis 1 & 2 are per-sample (trace vs that sample's trajectory / the scene); average.
    cons_vals, grnd_vals = [], []
    for i, trace in enumerate(rec.traces):
        if i < len(rec.trajectories):
            c = trace_consistency(trace.claims, rec.trajectories[i])
            if c is not None:
                cons_vals.append(c)
        g = trace_grounding(trace.claims, rec.scene_objects)
        if g is not None:
            grnd_vals.append(g)

    consistency = sum(cons_vals)/len(cons_vals) if cons_vals else None
    grounding = sum(grnd_vals)/len(grnd_vals) if grnd_vals else None

    stab = stability(rec.traces)

    return ClipScore(
        clip_id=rec.clip_id,
        consistency=consistency,
        grounding=grounding,
        stability=stab["score"],
        contradiction=stab["contradiction"],
        min_ade=rec.min_ade,
        n_samples=len(rec.traces),
    )


@dataclass
class DatasetScorecard:
    clip_scores: List[ClipScore] = field(default_factory=list)

    def _mean(self, attr):
        vals = [getattr(s, attr) for s in self.clip_scores]
        vals = [v for v in vals if v is not None]
        return sum(vals)/len(vals) if vals else None

    def summary(self) -> Dict:
        faiths = [s.faithfulness() for s in self.clip_scores]
        faiths = [f for f in faiths if f is not None]
        return {
            "n_clips": len(self.clip_scores),
            "faithfulness_mean": (sum(faiths)/len(faiths) if faiths else None),
            "consistency_mean": self._mean("consistency"),
            "grounding_mean": self._mean("grounding"),
            "stability_mean": self._mean("stability"),
            "contradiction_rate": (sum(1 for s in self.clip_scores if s.contradiction)
                                   / len(self.clip_scores) if self.clip_scores else None),
            "min_ade_mean": self._mean("min_ade"),
        }

    def format_table(self) -> str:
        lines = ["=" * 72,
                 f"{'clip':<14} | {'faith':>6} | {'cons':>5} {'grnd':>5} {'stab':>5} | {'contr':>5} | {'mADE':>5}",
                 "-" * 72]
        for s in self.clip_scores:
            f = s.faithfulness()
            lines.append(
                f"{s.clip_id[:14]:<14} | {('%.2f'%f) if f is not None else '  n/a':>6} | "
                f"{('%.2f'%s.consistency) if s.consistency is not None else ' n/a':>5} "
                f"{('%.2f'%s.grounding) if s.grounding is not None else ' n/a':>5} "
                f"{('%.2f'%s.stability) if s.stability is not None else ' n/a':>5} | "
                f"{'YES' if s.contradiction else '  .':>5} | "
                f"{('%.2f'%s.min_ade) if s.min_ade is not None else ' n/a':>5}")
        lines.append("=" * 72)
        summ = self.summary()
        fm = summ["faithfulness_mean"]
        lines.append(f"faithfulness mean: {('%.3f'%fm) if fm is not None else 'n/a'} "
                     f"over {summ['n_clips']} clips "
                     f"(faithfulness = mean of consistency/grounding/stability; minADE is accuracy, reported separately)")
        return "\n".join(lines)
