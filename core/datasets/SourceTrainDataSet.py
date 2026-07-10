import os
import numpy as np
from torch.utils import data
import pickle


class SourceTrainDataSet(data.Dataset):
    def __init__(self,
                 data_root,
                 data_list,
                 label_info,
                 max_iters=None,
                 num_classes=0,
                 split="train",
                 transform=None,
                 ignore_label=255,
                 debug=False):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        if max_iters is not None:
            self.file_label_pixel_count_all, self.image_scores = pickle.load(open(label_info, "rb"))
            files, scores = zip(*self.image_scores.items())
            scores = np.array(scores)

            if scores.sum() == 0:
                scores = np.ones_like(scores) / len(scores)
            else:
                scores = scores / scores.sum()

            files = list(files)

            shuffle_files = files.copy()
            np.random.shuffle(shuffle_files)
            initial_ids = shuffle_files.copy()

            remaining = max(0, max_iters - len(initial_ids))
            sampled_ids = list(np.random.choice(files, size=remaining, replace=True, p=scores))

            self.img_ids = initial_ids + sampled_ids

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, self.split, "images", name),
                    "label": os.path.join(self.data_root, self.split, "labels", name),
                    "name": os.path.normpath(name).split(os.sep)[1],
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
        name = datafiles["name"]

        if self.transform is not None:
            image, label = self.transform(image, label)

        ret_data = {
            "img": image,  # The transformed image
            'label': label,  # Transform and label mapped labels
            'index': index,  # Serial number
            'datafiles': datafiles,  # The address and other information of this picture
            'name': name,  # Image Name
        }

        return ret_data

