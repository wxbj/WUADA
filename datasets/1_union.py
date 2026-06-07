import os
import shutil

sub_types = ['images', 'labels']


def merge_val_to_train(base_path, modality=''):
    train_root = os.path.join(base_path, modality, 'train')
    val_root = os.path.join(base_path, modality, 'val')

    if not os.path.exists(val_root):
        return

    if modality == 'bSSFP':
        shutil.rmtree(val_root)
        return

    for sub_type in sub_types:
        src_parent_dir = os.path.join(val_root, sub_type)
        dst_parent_dir = os.path.join(train_root, sub_type)

        os.makedirs(dst_parent_dir, exist_ok=True)

        for item_name in os.listdir(src_parent_dir):
            src_path = os.path.join(src_parent_dir, item_name)
            dst_path = os.path.join(dst_parent_dir, item_name)
            shutil.move(src_path, dst_path)

    shutil.rmtree(val_root)


if __name__ == "__main__":
    merge_val_to_train('MM-WHS', 'ct')
    merge_val_to_train('MM-WHS', 'mr')
    merge_val_to_train('MS-CMRSeg', 'bSSFP')
    merge_val_to_train('MS-CMRSeg', 'LGE')

