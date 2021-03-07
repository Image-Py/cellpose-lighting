from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import cell_datasets


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = cell_datasets.CellDataset(data_dir=self.datadir, split=0.1)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CellDataLoader(BaseDataLoader):
    """
    Cell data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = cell_datasets.CellDataset(data_dir=self.data_dir, split='train2017', train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)