import torch

from herp.allocator import AllocationConfig, allocate_budget, priority_distribution
from herp.archive import RegionArchive, Snapshot
from herp.regions import OnlineRegionizer, RegionizerConfig
from herp.relevance import RelevanceTracker, cosine_relevance
from herp.sigma import branch_decomposition_sigma, pairwise_sigma


def test_pairwise_sigma_is_zero_for_identical_futures():
    z = torch.ones(4, 3)
    assert pairwise_sigma(z).item() < 1e-3


def test_pairwise_sigma_increases_with_future_dispersion():
    low = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    high = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    assert pairwise_sigma(high) > pairwise_sigma(low)


def test_branch_decomposition_separates_action_and_dyn_variance():
    z = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]])
    total, branch, dyn = branch_decomposition_sigma(z, lambda_dyn=0.0)
    assert branch.item() > 0.9
    assert dyn.item() < 1e-3
    assert torch.allclose(total, branch, atol=1e-5)


def test_regionizer_creates_and_reuses_radius_regions():
    archive = RegionArchive(max_snapshots_per_region=2, seed=0)
    regionizer = OnlineRegionizer(2, archive, RegionizerConfig(region_radius=1.0, max_regions=4))
    r0 = regionizer.assign(torch.tensor([0.0, 0.0]), step=0)
    r1 = regionizer.assign(torch.tensor([0.05, 0.0]), step=1)
    r2 = regionizer.assign(torch.tensor([10.0, 0.0]), step=2)
    assert r0 == r1
    assert r2 != r0
    assert len(archive) == 2


def test_archive_bounds_snapshots_with_reservoir_sampling():
    archive = RegionArchive(max_snapshots_per_region=3, seed=0)
    region = archive.add_region(torch.zeros(2), step=0)
    for i in range(20):
        archive.add_snapshot(region.region_id, Snapshot({}, torch.zeros(2), i, 0, 0.0))
    assert len(region.snapshots) == 3
    assert region.count == 21


def test_cosine_relevance_and_priority_distribution():
    archive = RegionArchive(seed=0)
    r0 = archive.add_region(torch.zeros(2), step=0)
    r1 = archive.add_region(torch.ones(2), step=0)
    tracker = RelevanceTracker()
    tracker.update_cosine(r0, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    tracker.update_cosine(r1, torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]))
    r0.sigma_ema = 2.0
    r1.sigma_ema = 0.1
    q = priority_distribution([r0, r1], AllocationConfig(uniform_mix=0.0))
    assert cosine_relevance(torch.tensor([1.0]), torch.tensor([1.0])) == 1.0
    assert q[0] > q[1]


def test_allocate_budget_conserves_total():
    archive = RegionArchive(seed=0)
    regions = [archive.add_region(torch.ones(2) * i, step=0) for i in range(3)]
    for i, region in enumerate(regions):
        region.p_ema = i + 1.0
        region.sigma_ema = i + 1.0
    allocation = allocate_budget(regions, 17, AllocationConfig(n_min=1), torch.Generator().manual_seed(1))
    assert sum(allocation.values()) == 17
    assert set(allocation) == {0, 1, 2}
