import pytest

torch = pytest.importorskip("torch")

from ct_ots_u.transport.datasets import TransportDataset


def test_transport_dataset_collate_multi_sample():
    h0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    h1 = h0 + 1
    cond = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dose = torch.tensor([0.5, 1.5], dtype=torch.float32)
    group = torch.tensor([1, 2], dtype=torch.long)
    batch = torch.tensor([0, 1], dtype=torch.long)

    dataset = TransportDataset(
        h0,
        h1,
        cond,
        dose=dose,
        group=group,
        batch=batch,
    )
    stacked = TransportDataset.collate([dataset[i] for i in range(len(dataset))])

    assert torch.allclose(stacked.h0, h0)
    assert torch.allclose(stacked.h1, h1)
    assert torch.allclose(stacked.cond, cond)
    assert stacked.dose is not None and torch.allclose(stacked.dose, dose)
    assert stacked.group is not None and torch.equal(stacked.group, group)
    assert stacked.batch is not None and torch.equal(stacked.batch, batch)

    no_meta = TransportDataset(h0, h1, cond)
    stacked_no_meta = TransportDataset.collate(
        [no_meta[i] for i in range(len(no_meta))]
    )
    assert stacked_no_meta.dose is None
    assert stacked_no_meta.group is None
    assert stacked_no_meta.batch is None
