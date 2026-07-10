import os
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool


def extract_label_info(labdir, NUM_CLASS):
    file_label_pixel_count_all = {}
    class_total_pixels = np.zeros(NUM_CLASS)

    for subfolder in os.listdir(labdir):
        lab_subdir = os.path.join(labdir, subfolder)
        for fname in os.listdir(lab_subdir):
            full_lab_path = os.path.join(lab_subdir, fname)
            relative_path = os.path.join(subfolder, fname)

            label_img = np.load(full_lab_path)
            labels, counts = np.unique(label_img, return_counts=True)

            pix_dict = {}
            for lab, count in zip(labels, counts):
                if 0 <= lab < NUM_CLASS:
                    pix_dict[lab] = int(count)
                    class_total_pixels[lab] += count

            file_label_pixel_count_all[relative_path] = pix_dict

    return file_label_pixel_count_all, class_total_pixels


def extract_label_info_worker(labdir, subfolder, fname, NUM_CLASS):
    full_lab_path = os.path.join(labdir, subfolder, fname)
    relative_path = os.path.join(subfolder, fname)

    label_img = np.load(full_lab_path)
    labels, counts = np.unique(label_img, return_counts=True)

    pix_dict = {}
    class_total_pixels = np.zeros(NUM_CLASS)

    for lab, count in zip(labels, counts):
        if 0 <= lab < NUM_CLASS:
            pix_dict[lab] = int(count)
            class_total_pixels[lab] += count

    return relative_path, pix_dict, class_total_pixels


def generate(labdir, out_file, nprocs=1, NUM_CLASS=4):
    if nprocs == 1:
        file_label_pixel_count_all, class_total_pixels = extract_label_info(labdir, NUM_CLASS)
    else:
        tasks = []
        for subfolder in os.listdir(labdir):
            for fname in os.listdir(os.path.join(labdir, subfolder)):
                tasks.append((labdir, subfolder, fname, NUM_CLASS))
        with Pool(nprocs) as pool:
            results = list(tqdm(pool.starmap(extract_label_info_worker, tasks), total=len(tasks)))

        file_label_pixel_count_all = {}
        class_total_pixels = np.zeros(NUM_CLASS)
        for rel_path, pix_dict, cls_pixels in results:
            file_label_pixel_count_all[rel_path] = pix_dict
            class_total_pixels += cls_pixels

    class_importance = 1.0 / np.log(1.1 + class_total_pixels)
    class_importance[0] = 0
    class_importance /= class_importance.sum()

    image_scores = {}
    for file, pixdict in file_label_pixel_count_all.items():
        if len(pixdict) == 0 or (len(pixdict) == 1 and 0 in pixdict):
            score = 0.1
        else:
            score = 1.0
        image_scores[file] = score
    with open(out_file, 'wb') as f:
        pickle.dump((file_label_pixel_count_all, image_scores), f)


def main():
    # MM-WHS
    generate('MM-WHS/ct/train/labels', 'MM-WHS/ct/ct_label_info.p', nprocs=8, NUM_CLASS=5)
    generate('MM-WHS/mr/train/labels', 'MM-WHS/mr/mr_label_info.p', nprocs=8, NUM_CLASS=5)
    # MS-CMRSeg
    generate('MS-CMRSeg/bSSFP/train/labels', 'MS-CMRSeg/bSSFP/bSSFP_label_info.p', nprocs=8, NUM_CLASS=4)
    generate('MS-CMRSeg/LGE/train/labels', 'MS-CMRSeg/LGE/LGE_label_info.p', nprocs=8, NUM_CLASS=4)

if __name__ == "__main__":
    main()

