import os
import numpy as np
import torch
from torch.utils import data


class OtherDataSet(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=0,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
            cfg=None,
            empty=False,
    ):
        self.active = True if split == 'active' else False
        if split == 'active':
            split = 'train'
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.cfg = cfg
        self.empty = empty
        with open(data_list, "r") as handle:
            content = [line.strip() for line in handle.readlines()]
        self.data_list = []
        if empty:
            self.data_list.append(
                {
                    "img": "",
                    "label": "",
                    "label_mask": "",
                    "name": "",
                }
            )
        else:
            for name in content:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root, self.split, 'images', name),
                        "label": os.path.join(
                            self.data_root, self.split, 'labels', name),
                        "label_mask": os.path.join(
                            cfg.OUTPUT_DIR, "gtMask", os.path.normpath(name).split(os.sep)[0],
                            f"{os.path.normpath(name).split(os.sep)[1].split('.npy')[0] + '_gtFine_labelIds.npy'}", ),
                        "name": os.path.normpath(name).split(os.sep)[1],
                        'indicator': os.path.join(cfg.OUTPUT_DIR, "gtIndicator",
                                                  os.path.normpath(name).split(os.sep)[0],
                                                  f"{os.path.normpath(name).split(os.sep)[1].split('.npy')[0] + '_indicator.pth'}"),
                        "path": os.path.normpath(name).split(os.sep)[0]
                    }
                )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = np.load(datafiles["img"]).astype(np.float32)
        label = np.load(datafiles["label"]).astype(np.uint8)

        if self.split == 'train':
            label_mask = np.load(datafiles["label_mask"]).astype(np.uint8)
        else:
            label_mask = np.ones_like(label, dtype=np.uint8) * 255

        origin_mask = torch.from_numpy(label_mask).long()

        active_indicator = torch.tensor([0])
        active_selected = torch.tensor([0])

        if self.active:
            indicator = torch.load(datafiles['indicator'])
            active_indicator = indicator['active']
            active_selected = indicator['selected']

            if active_indicator.size() == (1,):
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        origin_label = torch.from_numpy(label).long()

        label = np.expand_dims(label, axis=2)
        label_mask = np.expand_dims(label_mask, axis=2)

        h, w = label.shape[0], label.shape[1]

        mask_aggregation = np.concatenate((label, label_mask), axis=2)

        if self.transform is not None:
            image, mask_aggregation = self.transform(image, mask_aggregation)
            label = mask_aggregation[:, :, 0]
            label_mask = mask_aggregation[:, :, 1]

        ret_data = {
            "img": image,  # The transformed image
            'label': label,  # The transformed label
            'mask': label_mask,  # Active learning mask after transformation
            'name': datafiles['name'],  # Image Name
            'path_to_mask': datafiles['label_mask'],  # Mask Address
            'path_to_indicator': datafiles['indicator'],  # Indicator address
            'size': torch.tensor([h, w]),  # Used to interpolate the output to its original size
            'origin_mask': origin_mask,  # Untransformed mask
            'origin_label': origin_label,  # Untransformed labels
            'active': active_indicator,  # Active learning indicator
            'selected': active_selected,  # Active learning selector
            'path': datafiles['path']
        }

        return ret_data

