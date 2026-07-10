import os
import shutil
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def split(base_path, modality='ct', sub_types=None, indices=None):
    train_root = os.path.join(base_path, modality, 'train')
    val_root = os.path.join(base_path, modality, 'val')

    ref_dir = os.path.join(train_root, 'images')

    all_patients = sorted([d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d))],
                          key=natural_sort_key)

    target_patients = [all_patients[idx] for idx in indices]

    is_copy_operation = (modality == 'bSSFP')

    for sub_type in sub_types:
        src_parent_dir = os.path.join(train_root, sub_type)
        dst_parent_dir = os.path.join(val_root, sub_type)

        os.makedirs(dst_parent_dir, exist_ok=True)

        for patient_id in target_patients:
            src_path = os.path.join(src_parent_dir, patient_id)
            dst_path = os.path.join(dst_parent_dir, patient_id)

            if is_copy_operation:
                shutil.copytree(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)


if __name__ == "__main__":
    split('MM-WHS', 'ct', ['images', 'labels'], indices=[0, 1, 2, 3])
    split('MM-WHS', 'mr', ['images', 'labels'], indices=[0, 1, 2, 3])
    split('MS-CMRSeg', 'bSSFP', ['images', 'labels'], indices=[0, 1, 2, 3, 4])
    split('MS-CMRSeg', 'LGE', ['images', 'labels'], indices=[0, 1, 2, 3, 4])

    # split('MM-WHS', 'ct', ['images', 'labels'], indices=[4, 5, 6, 7])
    # split('MM-WHS', 'mr', ['images', 'labels'], indices=[4, 5, 6, 7])
    # split('MS-CMRSeg', 'bSSFP', ['images', 'labels'], indices=[5, 6, 7, 8, 9])
    # split('MS-CMRSeg', 'LGE', ['images', 'labels'], indices=[5, 6, 7, 8, 9])

    # split('MM-WHS', 'ct', ['images', 'labels'], indices=[8, 9, 10, 11])
    # split('MM-WHS', 'mr', ['images', 'labels'], indices=[8, 9, 10, 11])
    # split('MS-CMRSeg', 'bSSFP', ['images', 'labels'], indices=[10, 11, 12, 13, 14])
    # split('MS-CMRSeg', 'LGE', ['images', 'labels'], indices=[10, 11, 12, 13, 14])

    # split('MM-WHS', 'ct', ['images', 'labels'], indices=[12, 13, 14, 15])
    # split('MM-WHS', 'mr', ['images', 'labels'], indices=[12, 13, 14, 15])
    # split('MS-CMRSeg', 'bSSFP', ['images', 'labels'], indices=[15, 16, 17, 18, 19])
    # split('MS-CMRSeg', 'LGE', ['images', 'labels'], indices=[15, 16, 17, 18, 19])


