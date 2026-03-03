from . import transformer
from . import data
from . import training
from . import metrics
from . import visualization

# ── 强兼 Bridge Exports ──────────────────────────────────────────
# The following classes are bridged from xai-org/grok-1's architecture,
# re-implemented in PyTorch for grokking experiments.
from .transformer import (
    Transformer,            # Original OpenAI architecture
    GrokOneTransformer,     # 强兼: Grok-1 architecture (MoE + RoPE)
    GrokOneMoELayer,        # 强兼: Mixture of Experts layer
    GrokOneRouter,          # 强兼: Expert routing
    RotaryPositionalEmbedding,  # 强兼: RoPE from Grok-1
    RMSNorm,                # 强兼: RMS LayerNorm from Grok-1
)

__version__ = "0.0.2-qiangjian"
__bridge__ = "openai/grok ←→ xai-org/grok-1"
