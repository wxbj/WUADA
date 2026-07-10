import os
import glob
from tqdm import tqdm
import numpy as np


def generate_image_list(base_dir, output_file):
    output_list = [
        os.path.relpath(f, base_dir)
        for f in glob.glob(os.path.join(base_dir, '**', '*.npy'), recursive=True)
    ]
    with open(output_file, 'w') as f:
        f.writelines(path + '\n' for path in output_list)


def compute_mean_std_rgb(image_dir):
    image_paths = [
        path for path in glob.glob(os.path.join(image_dir, '**', '*.npy'), recursive=True)
        if 'label' not in path.lower() and 'conf' not in path.lower()
    ]
    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)
    pixel_count = 0

    for path in tqdm(image_paths, desc="Computing RGB mean/std"):
        img_np = np.load(path).astype(np.float32)

        flat = img_np.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_squared_sum += np.square(flat).sum(axis=0)
        pixel_count += img_np.shape[0] * img_np.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_squared_sum / pixel_count - mean ** 2)

    print(f"{image_dir}:")
    print(f"  mean = [{np.round(mean, 3).tolist()[0]:.3f},"
          f"{np.round(mean, 3).tolist()[1]:.3f},"
          f"{np.round(mean, 3).tolist()[2]:.3f}]")
    print(f"  std  = [{np.round(std, 3).tolist()[0]:.3f},"
          f"{np.round(std, 3).tolist()[1]:.3f},"
          f"{np.round(std, 3).tolist()[2]:.3f}]")

    return mean, std


def main():
    # MM-WHS
    generate_image_list('MM-WHS/ct/train/images', 'MM-WHS/ct/ct_train_list.txt')
    generate_image_list('MM-WHS/ct/test/images', 'MM-WHS/ct/ct_test_list.txt')
    generate_image_list('MM-WHS/ct/val/images', 'MM-WHS/ct/ct_val_list.txt')
    generate_image_list('MM-WHS/mr/train/images', 'MM-WHS/mr/mr_train_list.txt')
    generate_image_list('MM-WHS/mr/test/images', 'MM-WHS/mr/mr_test_list.txt')
    generate_image_list('MM-WHS/mr/val/images', 'MM-WHS/mr/mr_val_list.txt')
    compute_mean_std_rgb('MM-WHS/ct')
    compute_mean_std_rgb('MM-WHS/mr')
    # MS-CMRSeg
    generate_image_list('MS-CMRSeg/bSSFP/train/images', 'MS-CMRSeg/bSSFP/bSSFP_train_list.txt')
    generate_image_list('MS-CMRSeg/bSSFP/val/images', 'MS-CMRSeg/bSSFP/bSSFP_val_list.txt')
    generate_image_list('MS-CMRSeg/bSSFP/test/images', 'MS-CMRSeg/bSSFP/bSSFP_test_list.txt')
    generate_image_list('MS-CMRSeg/LGE/train/images', 'MS-CMRSeg/LGE/LGE_train_list.txt')
    generate_image_list('MS-CMRSeg/LGE/test/images', 'MS-CMRSeg/LGE/LGE_test_list.txt')
    generate_image_list('MS-CMRSeg/LGE/val/images', 'MS-CMRSeg/LGE/LGE_val_list.txt')
    compute_mean_std_rgb('MS-CMRSeg/bSSFP')
    compute_mean_std_rgb('MS-CMRSeg/LGE')



if __name__ == '__main__':
    main()
# MM-WHS/ct:
#   mean = [-1.067,-1.068,-1.069]
#   std  = [1.134,1.134,1.133]
# MM-WHS/mr:
#   mean = [0.069,0.069,0.068]
#   std  = [1.016,1.015,1.015]
# MS-CMRSeg/bSSFP:
#   mean = [-0.023,-0.024,-0.025]
#   std  = [1.001,1.000,1.000]
# MS-CMRSeg/LGE:
#   mean = [-0.023,-0.021,-0.020]
#   std  = [1.011,1.011,1.011]
