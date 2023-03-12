from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle = shuffle
