"""causal_ai_av — DevKit for AV Causal annotations on the Physical AI AV Dataset."""

from causal_ai_av.spec import AnnotationBundle

# `dataset` depends on `physical_ai_av`, which is gated behind the `[hf]`
# extra. Import lazily so this top-level package stays usable when the
# extra is not installed.
try:
    from causal_ai_av.dataset import CausalAVDataset, Sequence
except ImportError:  # pragma: no cover — exercised only without `[hf]`
    CausalAVDataset = None  # type: ignore[assignment]
    Sequence = None  # type: ignore[assignment]

__all__ = ["AnnotationBundle", "CausalAVDataset", "Sequence"]
