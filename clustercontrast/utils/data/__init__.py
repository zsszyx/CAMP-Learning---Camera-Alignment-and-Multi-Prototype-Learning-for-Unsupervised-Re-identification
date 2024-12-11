from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None
        self.count=0

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def __next__(self):
        if self.count >= self.length:
            self.count = 0
            raise StopIteration
        else:
            self.count+=1
            try:
                return next(self.iter)
            except:
                self.iter = iter(self.loader)
                return next(self.iter)

    def __iter__(self):
        return self
