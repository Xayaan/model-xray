import torch

from xray.train import ShuffleBuffer, lr_at


def test_buffer_trains_on_full_batches_even_when_passes_are_small():
    buf = ShuffleBuffer(capacity=64, batch=16, paired=True)
    seen = 0
    for i in range(20):  # each "forward pass" gives only 10 rows, fewer than one batch
        x = torch.full((10, 3), float(i))
        buf.add({"site": (x, x + 100)})
        for batch in buf.drain():
            bx, by = batch["site"]
            assert bx.shape == (16, 3) and torch.equal(by, bx + 100)  # pairs stay aligned
            seen += 16
    assert seen >= 200 - 64  # everything except what is still mixing in the buffer
    assert 32 <= len(buf) < 64 + 10


def test_unpaired_buffer_returns_none_targets():
    buf = ShuffleBuffer(capacity=8, batch=4, paired=False)
    x = torch.arange(24.0).reshape(12, 2)
    buf.add({"s": (x, x)})
    batches = list(buf.drain())
    assert batches and all(b["s"][1] is None for b in batches)


def test_lr_schedule_warms_up_and_decays():
    assert lr_at(0, 100, 1.0, warmup=10, decay_frac=0.2) == 0.1
    assert lr_at(50, 100, 1.0, warmup=10, decay_frac=0.2) == 1.0
    assert lr_at(99, 100, 1.0, warmup=10, decay_frac=0.2) < 0.1
