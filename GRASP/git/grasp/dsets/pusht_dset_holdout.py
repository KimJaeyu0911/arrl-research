"""PushT loader that holds out a fixed random subset of the *training* rollouts as extra validation data.

Nothing on disk is moved or modified. The split is done by rollout index and is fully determined by
`holdout_seed`, so the same held-out rollouts are selected on every call.

Use it without touching existing code, purely through hydra overrides, e.g.:

    python train.py env=pusht ... \
        env.dataset._target_=dsets.pusht_dset_holdout.load_pusht_slice_train_val_holdout \
        +env.dataset.n_holdout=1000 +env.dataset.holdout_seed=0
"""
import numpy as np
from .pusht_dset import PushTDataset
from .traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset


class ConcatTrajDataset(TrajDataset):
    """Concatenation of several trajectory datasets (possibly stored in different folders)."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def _locate(self, idx):
        if idx < 0:
            idx += len(self)
        k = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return self.datasets[k], idx - int(self.offsets[k])

    def __len__(self):
        return int(self.offsets[-1])

    def get_seq_length(self, idx):
        d, i = self._locate(idx)
        return d.get_seq_length(i)

    def __getitem__(self, idx, start=0, end=None, frameskip=1):
        d, i = self._locate(idx)
        return d.__getitem__(i, start, end, frameskip)

    def __getattr__(self, name):
        # only called for attributes not found normally: forward stats/dims (action_dim, action_mean, ...)
        if name in ("datasets", "offsets"):
            raise AttributeError(name)
        return getattr(self.datasets[0], name)


def load_pusht_slice_train_val_holdout(
    transform,
    n_rollout=None,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,  # unused, kept for signature compatibility with load_pusht_slice_train_val
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    n_holdout=1000,
    holdout_seed=0,
    include_orig_val=True,
    traj_valid_size=0,
):
    """Same return structure as `load_pusht_slice_train_val`: (datasets, traj_dset) dicts.

    n_holdout:        number of rollouts taken out of `<data_path>/train` and used only for validation.
    holdout_seed:     seed of the (deterministic) random choice of held-out rollouts.
    include_orig_val: also keep the original `<data_path>/val` rollouts in the validation set.
    traj_valid_size:  how many held-out rollouts are added to `traj_dset["valid"]`. That set is what plan.py
                      samples start/goal segments from, so the default 0 keeps it identical to the original
                      21 val rollouts (planning results stay comparable with earlier runs). The enlarged
                      validation set is always used for the validation-loss slices in `datasets["valid"]`.
                      None = all held-out rollouts (plan.py then loads every one of them to check lengths).
    """
    train_full = PushTDataset(
        n_rollout=n_rollout, transform=transform, data_path=data_path + "/train",
        normalize_action=normalize_action, with_velocity=with_velocity,
    )
    n = len(train_full)
    assert 0 < n_holdout < n, f"n_holdout must be in (0, {n})"
    holdout = np.sort(np.random.RandomState(holdout_seed).choice(n, n_holdout, replace=False))
    is_holdout = np.zeros(n, dtype=bool)
    is_holdout[holdout] = True
    train_idx = np.nonzero(~is_holdout)[0]
    assert not (set(train_idx.tolist()) & set(holdout.tolist()))

    train_dset = TrajSubset(train_full, train_idx.tolist())
    hold_dset = TrajSubset(train_full, holdout.tolist())
    parts_all, parts_traj = [hold_dset], []
    k = len(holdout) if traj_valid_size is None else int(traj_valid_size)
    if k > 0:
        parts_traj.append(TrajSubset(train_full, holdout[:k].tolist()))
    if include_orig_val:
        val_orig = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=data_path + "/val",
            normalize_action=normalize_action, with_velocity=with_velocity,
        )
        parts_all.append(val_orig)
        parts_traj.append(val_orig)
    assert parts_traj, "traj_dset['valid'] would be empty (traj_valid_size=0 needs include_orig_val=True)"
    val_dset = ConcatTrajDataset(parts_all)
    val_traj_dset = ConcatTrajDataset(parts_traj)
    print(
        f"[holdout] train rollouts: {len(train_dset)} | validation rollouts: {len(val_dset)} "
        f"(held-out from train: {len(hold_dset)}, orig val: {len(val_dset) - len(hold_dset)}) | "
        f"traj_dset['valid']: {len(val_traj_dset)} | first held-out idx: {holdout[:5].tolist()}"
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_traj_dset}
    return datasets, traj_dset
