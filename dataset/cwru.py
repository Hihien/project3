import os

import torch
from collections import namedtuple
from scipy.io import loadmat

__all__ = ['CWRUDataset']

CWRUSample = namedtuple('CWRUSample',
                        ('mat_id', 'partition', 'label', 'diameter', 'motor_load', 'position', 'rpm', 'slice_id'))


class CWRUDataset:
    def __init__(self,
                 root,
                 slice_dim=10000,
                 step_dim=None,
                 feature_type='DE',
                 ):
        self.root = root
        self.slice_dim = slice_dim
        self.step_dim = step_dim if step_dim is not None else slice_dim
        self.feature_type = feature_type

        self.samples = []
        self.targets = []
        self.infos = []
        self.classes = set()
        for root, dirs, files in os.walk(self.root):
            for f in filter(lambda _: _.endswith('.mat'), files):
                f = os.path.join(root, f).replace(os.sep, '/')
                tokens = f.split('/')

                position = None
                diameter = None
                if 'Normal' not in tokens[-1]:
                    partition_ind = -4
                    label_ind = -3
                    fault_diameter_ind = -2
                    if '@' in tokens[-2]:
                        partition_ind -= 1
                        label_ind -= 1
                        fault_diameter_ind -= 1
                        position = int(tokens[-2][1:])
                    if '@' in tokens[-1]:
                        position = int(tokens[-1][tokens[-1].find('@') + 1:tokens[-1].find('_')])
                        mat_id = int(tokens[-1][:tokens[-1].find('@')])
                    else:
                        mat_id = int(tokens[-1][:tokens[-1].find('_')])
                    partition = tokens[partition_ind]
                    label = tokens[label_ind]
                    diameter = int(tokens[fault_diameter_ind])
                else:
                    partition = 'Normal'
                    label = 'Normal'
                    mat_id = int(tokens[-1][:tokens[-1].find('_')])
                motor_load = int(tokens[-1][tokens[-1].rfind('_') + 1:tokens[-1].rfind('.')])

                mat = loadmat(f)
                prefix = f"X{mat_id:03d}"
                rpm = mat.get(prefix + "RPM")
                if rpm is not None:
                    rpm = rpm.item()

                self.classes.add(label)
                try:
                    feature = mat[f"{prefix}_{self.feature_type}_time"]
                except KeyError as e:
                    for k in mat.keys():
                        if feature_type in k:
                            feature = mat[k]
                            break
                    else:
                        raise e
                feature = torch.from_numpy(feature).float().squeeze(-1)

                start_inds = list(range(0, feature.shape[0] - self.slice_dim, self.step_dim))
                for slice_id, start_ind in enumerate(start_inds):
                    feature_slice = feature[start_ind:start_ind + self.slice_dim]
                    info = CWRUSample(mat_id, partition, label, diameter, motor_load, position, rpm, slice_id)
                    self.samples.append(feature_slice)
                    self.targets.append(label)
                    self.infos.append(info)
        self.classes = sorted(self.classes, key=lambda _: (len(_), _[0]))
        self.class_to_idx = lambda _: self.classes.index(_)
        self.targets = torch.tensor(list(map(self.class_to_idx, self.targets)), dtype=torch.long)
        self.samples = torch.stack(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.targets[item]
