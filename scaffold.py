"""
Tiny GPT From Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

import numpy as np


TOY_CORPUS = (
    "hello world\nthe quick brown fox jumps over the lazy dog\n"
    "tiny gpt learns characters one step at a time\n"
) * 20


def build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2):
    tok_emb = create_token_embedding(vocab_size, d_model)
    pos_emb = create_positional_embedding(block_size, d_model)
    blocks = stack_transformer_blocks(n_layers, d_model, n_heads, d_ff)
    return {
        "tok_emb": tok_emb, "pos_emb": pos_emb, "blocks": blocks,
        "ln_f": {"gamma": np.ones((d_model,)), "beta": np.zeros((d_model,))},
        "lm_head": {"w_lm": np.random.randn(d_model, vocab_size) * 0.02,
                    "b_lm": np.zeros((vocab_size,))},
        "block_size": block_size, "vocab_size": vocab_size,
    }


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # 1) Tokenizer + corpus prep
    text = read_text_file(TOY_CORPUS)
    vocab = build_vocab(text)
    stoi = build_stoi(vocab)
    itos = build_itos(vocab)
    vocab_size = len(vocab)
    print(f"vocab_size={vocab_size}, vocab[:10]={vocab[:10]}")

    data = encode_corpus_to_int_array(text, stoi)
    split_idx = pick_split_point(len(data), 0.9)
    train_ids, val_ids = slice_train_and_val(data, split_idx)
    print(f"train={len(train_ids)} val={len(val_ids)}")

    # 2) Batch sanity check
    block_size = pick_block_size(8)
    xb, yb = get_batch(train_ids, block_size, batch_size=4, rng=rng)
    print(f"batch X shape={xb.shape} Y shape={yb.shape}")

    # 3) Build the GPT model
    params = build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # 4) Train
    result = wire_full_training_loop(
        params, train_ids, block_size,
        batch_size=16, learning_rate=3e-3, num_steps=200, log_every=20,
    )
    print("loss_history:", result["loss_history"])

    val_loss = logging_and_validation_loss(result["params"], val_ids, block_size, 4, 2)
    print(f"val_loss ~ {val_loss:.4f}")

    # 5) Generate text from a prompt
    prompt_ids = encode_prompt("hello", stoi)
    generated = generation_loop_for_n_steps(
        result["params"], prompt_ids, 40, block_size,
        temperature=1.0, top_k=5, rng=rng,
    )
    print("generated:", repr(decode_final_sequence(generated, itos)))
