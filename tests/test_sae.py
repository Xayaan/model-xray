import torch

from xray.sae import TopKDictionary, sparse_decode


def synthetic(n_rows=4096, n_feats=64, d=32, active=3, seed=0):
    """Rows built from a few positive multiples of unit dictionary directions."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_feats, d, generator=g)
    D /= D.norm(dim=1, keepdim=True)
    idx = torch.stack([torch.randperm(n_feats, generator=g)[:active] for _ in range(n_rows)])
    vals = torch.rand(n_rows, active, generator=g) + 0.5
    codes = torch.zeros(n_rows, n_feats).scatter_(1, idx, vals)
    return codes, D


def fit(d: TopKDictionary, x, y, steps=1500, lr=2e-3):
    d.init_from_data(x, y)
    opt = torch.optim.Adam(d.parameters(), lr=lr)
    for step in range(steps):
        batch = torch.randint(0, x.shape[0], (256,))
        fvu, aux, _ = d.loss(x[batch], y[batch])
        opt.zero_grad()
        (fvu + aux / 32).backward()
        d.remove_parallel_grads()
        opt.step()
        d.normalize_decoder()
    with torch.no_grad():
        return float(d.loss(x, y)[0])


def test_encode_is_k_sparse():
    d = TopKDictionary(16, 64, 4)
    x = torch.randn(32, 16)
    vals, idx = d.encode(x)
    assert vals.shape == idx.shape == (32, 4)
    assert bool(((d.dense_codes(x) > 0).sum(-1) <= 4).all())


def test_sparse_decode_matches_dense():
    W = torch.randn(50, 8)
    idx = torch.randint(0, 50, (10, 5))
    vals = torch.rand(10, 5)
    dense = torch.zeros(10, 50).scatter_add_(1, idx, vals)
    assert torch.allclose(sparse_decode(vals, idx, W), dense @ W, atol=1e-5)


def test_folded_weights_match_module():
    torch.manual_seed(0)
    for transcoder in (False, True):
        d = TopKDictionary(12, 48, 5, d_out=10 if transcoder else None, transcoder=transcoder, skip=transcoder)
        x = torch.randn(64, 12) * 37
        y = torch.randn(64, 10) * 0.3 if transcoder else None
        d.init_from_data(x, y)
        with torch.no_grad():
            for p in d.parameters():
                p.add_(0.05 * torch.randn_like(p))
            f = d.folded()
            pre = (x if transcoder else x - f["b_dec"]) @ f["W_enc"] + f["b_enc"]
            vals, idx = pre.topk(5, dim=-1)
            out = torch.zeros(64, 48).scatter_(1, idx, vals.relu()) @ f["W_dec"] + f["b_dec"]
            if transcoder:
                out = out + x @ f["W_skip"]
            assert torch.allclose(out, d(x), rtol=1e-4, atol=1e-4)


def test_sae_recovers_sparse_structure():
    torch.manual_seed(0)
    codes, D = synthetic()
    x = codes @ D
    d = TopKDictionary(32, 128, 4)
    assert fit(d, x, x) < 0.1
    assert torch.allclose(d.W_dec.norm(dim=1), torch.ones(128), atol=1e-4)


def test_transcoder_learns_a_sparse_map():
    torch.manual_seed(0)
    codes, D_in = synthetic(seed=1)
    _, D_out = synthetic(seed=2, d=24)
    x, y = codes @ D_in, codes @ D_out
    d = TopKDictionary(32, 128, 4, d_out=24, transcoder=True, skip=True)
    assert fit(d, x, y) < 0.15


def test_save_and_load_roundtrip(tmp_path):
    d = TopKDictionary(8, 32, 3, d_out=6, transcoder=True, skip=True)
    x = torch.randn(16, 8)
    d.init_from_data(x, torch.randn(16, 6))
    d.save(tmp_path / "dict", extra={"site": {"name": "L0.mlp"}})
    loaded = TopKDictionary.load(tmp_path / "dict")
    assert torch.allclose(loaded(x), d(x))
