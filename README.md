# Tiny GPT From Scratch

Tiny GPT From Scratch is a character-level GPT built by hand in plain NumPy.
There is no PyTorch, no autograd, and no deep learning framework of any kind.
Every piece of a working Transformer is written out as one of 166 small, numbered steps: tokenizer, softmax, attention, backpropagation, Adam, and everything between.
The point is to see exactly how a GPT works by building each part yourself, instead of importing it.

A few steps are written derivations, not code.
Before a backward pass gets implemented, a `derive_..._on_paper` step returns the gradient worked out by hand — the algebra first, the NumPy after.
`derive_dlogits_on_paper` and `derive_dw_on_paper` are two examples, each returning a short written proof instead of a numeric result.

Nothing here is optimized for speed. It is optimized for reading: one function, one concept, one step at a time.

The steps started as a sequence of exercises on Deep-ML, solved one at a time.
This repository assembles them, in order, into one running project.

## Structure

The repository is small on purpose: two Python files, a test suite, and a docs page.

- `model.py` holds all 166 step functions, in the order you build them. Each one is small and does one job: a tokenizer helper, a NumPy primitive, one layer's forward or backward pass, an optimizer update.
  - Later steps call earlier ones — `get_batch`, for instance, calls `sample_random_batch_offsets`, `stack_x_batch`, and `stack_y_batch`.
- `scaffold.py` is a runnable demo that wires the finished steps together: tokenize a toy corpus, build a batch, build the model, train it with Adam, check validation loss, and generate text from a prompt.
- `tests/` holds the pytest suite, covering the steps end to end, including numerical gradient checks on the backward passes.
- `docs/` holds a small static project page that walks through the same eight parts. Open `docs/index.html` directly in a browser; nothing needs to be built or served.

## The eight parts

The steps build up in order, from raw characters to a full Transformer.
Each part below is a stretch of consecutive steps in `model.py`.

| Part | Steps | What it covers |
| --- | --- | --- |
| 1. Tokenizer | 1-7 | A vocabulary over the corpus, `stoi`/`itos` lookups, and encode/decode. |
| 2. NumPy and softmax foundations | 8-33 | Arrays, indexing, broadcasting, reductions, and a numerically stable softmax. |
| 3. Data pipeline and bigram baseline | 34-56 | Loading the corpus, splitting and batching it, and a counting bigram model. |
| 4. Single-layer neural bigram | 57-73 | A learned weight matrix in place of the count table, cross-entropy, gradients, and SGD. |
| 5. Layer primitives and backprop | 74-91 | Forward and backward passes for linear layers, bias, ReLU, softmax cross-entropy, and LayerNorm. |
| 6. Embeddings and self-attention | 92-130 | Token and positional embeddings, then masked single-head and multi-head attention, forward and backward. |
| 7. FFN, blocks, and full model | 131-146 | Feed-forward layers, residual connections, pre-LN Transformer blocks, and the full model forward and backward. |
| 8. Adam, training loop, and generation | 147-166 | The Adam optimizer, the training and validation loop, and temperature/top-k sampling. |

By the end, the same functions add up to a full GPT: token and positional embeddings, causal self-attention, feed-forward blocks, and a sampler that turns logits back into text.

## Running it

Requires Python 3, NumPy, and pytest. `pip install -r requirements.txt` installs both.

Run the demo:

```bash
python scaffold.py
```

A run looks like this:

```
vocab_size=28, vocab[:10]=['\n', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
train=1836 val=204
batch X shape=(4, 8) Y shape=(4, 8)
loss_history: [3.329, 2.798, 2.277, 1.7, 1.321, 1.063, 0.969, 0.794, 0.685, 0.564]
val_loss ~ 0.5464
generated: 'hello fox jump\nt ony gpchearazy ons juick bro'
```

The loss falls as it trains; the sample is rough because the toy corpus is tiny.

Run the test suite:

```bash
python -m pytest
```

The toy corpus lives in `TOY_CORPUS` inside `scaffold.py`.
The model size — `d_model`, `n_heads`, `d_ff`, `n_layers` — is set in the same file, in the `build_model` call.
Edit either one to try something bigger.

---

Built by **@Avi-Gobrin** on [Deep-ML](https://deep-ml.com).
