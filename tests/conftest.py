import numpy as np

from model import create_positional_embedding, create_token_embedding, stack_transformer_blocks

TINY_VOCAB = 11
TINY_BLOCK = 5
TINY_D_MODEL = 8
TINY_N_HEADS = 2
TINY_D_FF = 16
TINY_N_LAYERS = 2


def numerical_gradient(f, x, h=1e-6):
    """Central-difference gradient of a scalar-valued f at every element of x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + h
        plus = f(x)
        x[idx] = original - h
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * h)
    return grad


def make_tiny_params(seed=0):
    """Build a full GPT parameter tree at the fixed tiny test dimensions."""
    np.random.seed(seed)
    tok_emb = create_token_embedding(TINY_VOCAB, TINY_D_MODEL)
    pos_emb = create_positional_embedding(TINY_BLOCK, TINY_D_MODEL)
    blocks = stack_transformer_blocks(TINY_N_LAYERS, TINY_D_MODEL, TINY_N_HEADS, TINY_D_FF)
    return {
        'tok_emb': tok_emb,
        'pos_emb': pos_emb,
        'blocks': blocks,
        'ln_f': {'gamma': np.ones(TINY_D_MODEL), 'beta': np.zeros(TINY_D_MODEL)},
        'lm_head': {'w_lm': np.random.randn(TINY_D_MODEL, TINY_VOCAB) * 0.02,
                    'b_lm': np.zeros(TINY_VOCAB)},
    }
