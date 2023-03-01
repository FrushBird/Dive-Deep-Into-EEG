import os
import re
from copy import deepcopy
import numpy as np
import torch


class DatasetTemplate_train(torch.utils.data.Dataset):
    def __init__(self,):
        # Please download and unzip sample datas and set up your own path

        self.dataset_dir = r'Datas\train'
        _ = os.listdir(self.dataset_dir)
        fns = _.sort(key=lambda x: int(re.findall(r"\d+", x.split('+')[0])[0]))


        self.labels = []
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0]) - 1
            sample = torch.from_numpy(np.load(path))
            samples.append(sample)
            self.labels.append(ans)

        # [Nums,Cells,Features]
        self.samples = deepcopy(samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class DatasetTemplate_test(torch.utils.data.Dataset):
    def __init__(self, ):
        # Please download and unzip sample datas and set up your own path

        self.dataset_dir = r'Datas\test'
        _ = os.listdir(self.dataset_dir)
        fns = _.sort(key=lambda x: int(re.findall(r"\d+", x.split('+')[0])[0]))

        self.labels = []
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0]) - 1
            sample = torch.from_numpy(np.load(path))
            samples.append(sample)
            self.labels.append(ans)

        # [Nums,Cells,Features]
        self.samples = deepcopy(samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
