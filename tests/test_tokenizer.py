from model import build_itos, build_stoi, build_vocab, decode_ids, decode_int, encode_char, encode_string

TEXT = "the quick brown fox jumps over the lazy dog"


def test_vocab_is_sorted_and_unique():
    vocab = build_vocab(TEXT)
    assert vocab == sorted(vocab)
    assert len(vocab) == len(set(vocab))


def test_stoi_and_itos_are_inverses():
    vocab = build_vocab(TEXT)
    stoi = build_stoi(vocab)
    itos = build_itos(vocab)
    assert len(stoi) == len(vocab)
    assert len(itos) == len(vocab)
    assert all(itos[stoi[ch]] == ch for ch in vocab)


def test_encode_decode_string_round_trip():
    vocab = build_vocab(TEXT)
    stoi = build_stoi(vocab)
    itos = build_itos(vocab)
    assert decode_ids(encode_string(TEXT, stoi), itos) == TEXT


def test_encode_decode_single_char_round_trip():
    vocab = build_vocab(TEXT)
    stoi = build_stoi(vocab)
    itos = build_itos(vocab)
    for ch in vocab:
        assert decode_int(encode_char(ch, stoi), itos) == ch
