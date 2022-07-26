import json
import os
from datetime import datetime, timedelta

import numpy as np
import torch


def gen_log_folder_name(root="./", prefix=""):
    now = datetime.now()

    while os.path.exists(os.path.join(root, prefix+now.strftime("%m%d_%H%M"))):
        now += timedelta(minutes=1)

    return os.path.join(root, prefix+now.strftime("%m%d_%H%M"))


def dump_args(args, log_dir):
    args_dict = args.__dict__

    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)


def load_args(args, log_dir):
    with open(os.path.join(log_dir, 'args.json'), 'r') as fp:
        args_dict = json.load(fp)

    args.__dict__ = args_dict

    return args


class AvgMeter():

    def __init__(self, names: list):
        self.names = names
        self.clear()

    @torch.no_grad()
    def append(self, key, data):
        self.recoder[key].append(
            data.cpu().item() if torch.is_tensor(data) else data)

    def __len__(self):
        return len(self.recoder[self.names[0]])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.recoder[key]
        else:
            items = {}
            for name in self.names:
                if len(self.recoder[name]):
                    items[name] = self.recoder[name][key]
            return items

    def mean(self):
        items = {}
        for name in self.names:
            items[name] = np.mean(self.recoder[name]) if len(
                self.recoder[name]) else None

        return items

    def clear(self):
        self.recoder = {}
        for name in self.names:
            self.recoder[name] = []


class Queue():
    def __init__(self, max_size):
        self.max_size = max_size
        self._values = []

    def put(self, item):
        self._values.append(item)
        if len(self._values) > self.max_size:
            self._values.pop(0)

    def mean(self):
        return np.mean(self._values) if len(self._values) else np.zeros(1).mean()


if __name__ == '__main__':
    print(gen_log_folder_name())
