"""Factory functions for constructing paired mammogram data loaders."""

from torch.utils.data import DataLoader

from .private_dataset import PrivatePairedMammogramDataset


def build_dataloader(
    csv_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch DataLoader for paired prior-current mammograms."""
    dataset = PrivatePairedMammogramDataset(
        csv_path=csv_path,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
