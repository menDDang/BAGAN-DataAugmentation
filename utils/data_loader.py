import os
import numpy as np

commands = {"left":0, "down":1, "go":2, "off":3, "on":4, "right":5, "stop":6, "up":7, "yes":8, "no":9}
labels = ['backward', 'dog', 'follow', 'happy', 'marvin', 'on', 'sheila', 'tree', 'wow', 'bed', \
          'down', 'forward', 'house', 'nine', 'one', 'six', 'two', 'yes', 'bird', 'eight', \
          'four', 'learn', 'no', 'right', 'stop', 'up', 'zero', 'cat', 'five', 'go', \
          'left', 'off', 'seven', 'three', 'visual'
          ]


def get_data_in_command(mode, command, hp):
    if not mode in ['training', 'testing', 'validation']:
        raise ValueError("mode must be in 'training', 'testing', 'validation'")

    if not command in commands:
        raise ValueError("{} is not in commands".format(command))

    filename = os.path.join(hp.path.feat_dir, mode, command + ".npy")
    x = np.load(filename)
    return x


def get_data_in_noncommand(mode, hp):
    x = []
    for label in labels:
        if label in commands:
            continue

        filename = os.path.join(hp.path.feat_dir, mode, label + ".npy")
        _x = np.load(filename)
        x.append(_x)

    x = np.vstack(x)
    return x


class Dataset:
    def __init__(self, mode, hp, r=1):
        if not mode in ['training', 'testing', 'validation']:
            raise ValueError("mode must be in 'training', 'testing', 'validation'")

        self.hp = hp
        self.mode = mode
        self.commands = commands

        self.x = dict()
        for command in commands:
            _x = get_data_in_command(mode, command, hp)
            idx = int(len(_x) * r)
            self.x[command] = _x[:idx]
        self.x['unknown'] = get_data_in_noncommand(mode, hp)

        self.max_data_num = len(self.x['unknown'])

    def get_batch(self, batch_size):
        x, y = [], []
        for i in range(batch_size):
            for key, data in self.x.items():
                idx = np.random.randint(0, len(data))
                _x = data[idx]
                _y = np.zeros(shape=[11], dtype=np.float32)
                if key == 'unknown':
                    _y[-1] = 1
                else:
                    _y[commands[key]] = 1

                x.append(_x)
                y.append(_y)

        x = np.stack(x, axis=0)
        y = np.vstack(y)

        return x, y


class BAGANDataset:
    def __init__(self, mode, hp, r=1):
        if not mode in ['training', 'testing', 'validation']:
            raise ValueError("mode must be in 'training', 'testing', 'validation'")

        self.hp = hp
        self.mode = mode
        self.commands = commands

        self.x = dict()
        for command in commands:
            _x = get_data_in_command(mode, command, hp)
            idx = int(len(_x) * r)
            self.x[command] = _x[:idx]

    def get_batch(self, batch_size):
        x = []
        for i in range(batch_size):
            for key, data in self.x.items():
                idx = np.random.randint(0, len(data))
                _x = data[idx]

                x.append(_x)

        x = np.stack(x, axis=0)

        return x