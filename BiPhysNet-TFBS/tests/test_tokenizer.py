import torch

from biphysnet.tokenizer import KmerTokenizer, reverse_complement


def test_kmer_tokenizer_encode_shape_and_type():
    tokenizer = KmerTokenizer(kmerlen=6, maxlen=16)
    encoded = tokenizer.encode("ATCGATCGATCG")
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dtype == torch.long
    assert encoded.shape[0] == 16
    assert int(encoded[0]) == tokenizer.cls_token_id


def test_kmer_tokenizer_respects_max_length():
    tokenizer = KmerTokenizer(kmerlen=3, maxlen=8)
    encoded = tokenizer.encode("ATCGATCGATCGATCG")
    assert encoded.shape[0] == 8


def test_reverse_complement():
    assert reverse_complement("ATCGN") == "NCGAT"
