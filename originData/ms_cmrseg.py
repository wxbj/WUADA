import SimpleITK as sitk
import numpy as np
import os
import shutil
import re

data_folder = 'MS-CMRSeg'
bssfp_folder = 'bSSFP'
lge_folder = 'LGE'
size = 192
voxel = 1.0
save_folder = 'MS-CMRSeg'
view = 'axial'
keep_labels = [200, 500, 600]
label_mapping = {0: 0, 200: 1, 500: 2, 600: 3}


def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def get_nii_files(folder):
    image_data_files = []
    label_data_files = []
    for file_name in sorted(os.listdir(os.path.join(data_folder, folder)), key=numerical_sort_key):
        if file_name.endswith('C0.nii.gz') or file_name.endswith('LGE.nii.gz'):
            image_data_files.append({file_name: sitk.ReadImage(os.path.join(data_folder, folder, file_name))})
        elif file_name.endswith('manual.nii.gz'):
            label_path = os.path.join(data_folder, folder, file_name)
            label = sitk.Cast(sitk.ReadImage(label_path), sitk.sitkUInt16)
            label_data_files.append({file_name: label})
    return image_data_files, label_data_files


def resample_image(data_files, new_spacing=(voxel, voxel, voxel), is_label=False):
    def clip_high_percentile(img_np, upper_percent=0.0):
        upper_value = np.percentile(img_np, upper_percent)
        img_np = np.clip(img_np, None, upper_value)
        return img_np

    for data_file in data_files:
        for name, image in data_file.items():
            if not is_label:
                img_np = sitk.GetArrayFromImage(image)
                img_np = clip_high_percentile(img_np, upper_percent=98)
                image = sitk.GetImageFromArray(img_np)
                image.CopyInformation(data_file[name])

            original_spacing = image.GetSpacing()
            original_size = image.GetSize()

            new_size = [
                int(round(osz * ospc / nspc))
                for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
            ]

            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(new_spacing)
            resample.SetSize(new_size)
            resample.SetOutputOrigin(image.GetOrigin())
            resample.SetOutputDirection(image.GetDirection())

            resample.SetDefaultPixelValue(0)

            resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

            resampled_img = resample.Execute(image)
            data_file[name] = resampled_img
    return data_files


def crop_images_and_labels(image_data_files, label_data_files, crop_size=(size, size, size)):
    cropped_image_data_files = []
    cropped_label_data_files = []

    for image_entry, label_entry in zip(image_data_files, label_data_files):
        image_path, image = next(iter(image_entry.items()))
        label_path, label_img = next(iter(label_entry.items()))

        size = image.GetSize()

        max_dim = max(size)
        pad_lower = [max((max_dim - s) // 2, 0) for s in size]
        pad_upper = [max(max_dim - s - pl, 0) for s, pl in zip(size, pad_lower)]

        if any(p > 0 for p in pad_lower + pad_upper):
            image_pad = sitk.ConstantPadImageFilter()
            image_pad.SetPadLowerBound(pad_lower)
            image_pad.SetPadUpperBound(pad_upper)
            image_pad.SetConstant(0)
            image = image_pad.Execute(image)

            label_pad = sitk.ConstantPadImageFilter()
            label_pad.SetPadLowerBound(pad_lower)
            label_pad.SetPadUpperBound(pad_upper)
            label_pad.SetConstant(0)
            label_img = label_pad.Execute(label_img)

            size = image.GetSize()

        label_array = sitk.GetArrayFromImage(label_img)
        nonzero_idx = np.argwhere(label_array > 0)

        centroid_zyx = np.mean(nonzero_idx, axis=0)
        centroid_xyz = [int(round(c)) for c in centroid_zyx[::-1]]

        start = [max(0, c - cs // 2) for c, cs in zip(centroid_xyz, crop_size)]
        start = [min(s, sz - cs) for s, cs, sz in zip(start, crop_size, size)]

        roi = sitk.RegionOfInterestImageFilter()
        roi.SetSize(crop_size)
        roi.SetIndex(start)

        cropped_image = roi.Execute(image)
        cropped_label = roi.Execute(label_img)

        cropped_image_data_files.append({image_path: cropped_image})
        cropped_label_data_files.append({label_path: cropped_label})

    return cropped_image_data_files, cropped_label_data_files


def extract_all_view_slices(image_data_files, label_data_files, image_type):
    def clear_directory(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def get_slices_by_view(img_array):
        if view == 'axial':
            return [img_array[i, :, :] for i in range(img_array.shape[0])]
        elif view == 'coronal':
            return [img_array[:, i, :] for i in range(img_array.shape[1])]
        elif view == 'sagittal':
            return [img_array[:, :, i] for i in range(img_array.shape[2])]
        else:
            raise ValueError(f"Unsupported view: {view}")

    def compute_foreground_mean_std(volume):
        mask = (volume != 0)
        foreground = volume[mask]
        mean = np.mean(foreground)
        std = np.std(foreground)
        return mean, std

    def save_full_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix, volume_mean,
                             volume_std):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        save_idx = 0
        for i in range(1, len(image_slices) - 1):
            prev_img = image_slices[i - 1]
            curr_img = image_slices[i]
            next_img = image_slices[i + 1]

            image = np.stack([prev_img, curr_img, next_img], axis=-1)

            if any(np.all(image[:, :, ch] == 0) for ch in range(image.shape[-1])):
                continue
            if np.all(label_slices == 0):
                continue

            label = label_slices[i]

            ch = (image - volume_mean) / volume_std

            np.save(os.path.join(image_dir, f"{image_prefix}_{view}_{save_idx:03d}.npy"),
                    ch.astype(np.float32))
            np.save(os.path.join(label_dir, f"{label_prefix}_{view}_{save_idx:03d}.npy"),
                    label.astype(np.uint8))

            save_idx += 1

    def save_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix, volume_mean,
                        volume_std):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        save_idx = 0

        for i in range(1, len(image_slices) - 1):

            prev_img = image_slices[i - 1]
            curr_img = image_slices[i]
            next_img = image_slices[i + 1]

            image = np.stack([prev_img, curr_img, next_img], axis=-1)

            if any(np.all(image[:, :, ch] == 0) for ch in range(image.shape[-1])):
                continue

            label = label_slices[i]

            ch = (image - volume_mean) / volume_std

            np.save(os.path.join(image_dir, f"{image_prefix}_{view}_{save_idx:03d}.npy"),
                    ch.astype(np.float32))
            np.save(os.path.join(label_dir, f"{label_prefix}_{view}_{save_idx:03d}.npy"),
                    label.astype(np.uint8))

            save_idx += 1

    if image_type == "LGE":
        test_end = 5

    for idx, (image_entry, label_entry) in enumerate(zip(image_data_files, label_data_files)):
        image_path, image = next(iter(image_entry.items()))
        label_path, label_img = next(iter(label_entry.items()))

        if image_type == "bSSFP":
            phase = 'train'
        else:
            if idx < test_end:
                phase = 'test'
            else:
                phase = 'train'

        base_dir = os.path.join('..', 'datasets', save_folder, image_type, phase)
        if image_type == "bSSFP":
            label_prefix = os.path.basename(label_path)[:-17]
            image_prefix = os.path.basename(image_path)[:-10]
        else:
            label_prefix = os.path.basename(label_path)[:-18]
            image_prefix = os.path.basename(image_path)[:-11]
        label_dir = os.path.join(base_dir, 'labels', label_prefix)
        image_dir = os.path.join(base_dir, 'images', image_prefix)

        clear_directory(label_dir)
        clear_directory(image_dir)

        oriented_label_array = sitk.GetArrayFromImage(label_img)
        oriented_image_array = sitk.GetArrayFromImage(image)

        volume_mean, volume_std = compute_foreground_mean_std(oriented_image_array)

        label_slices = get_slices_by_view(oriented_label_array.copy())
        image_slices = get_slices_by_view(oriented_image_array.copy())

        for i in range(len(label_slices)):
            label = label_slices[i]
            new_label = np.zeros_like(label)
            for src_val, dst_val in label_mapping.items():
                new_label[label == src_val] = dst_val
            label_slices[i] = new_label
        if phase == "test":
            save_full_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix,
                                 volume_mean,
                                 volume_std)
        else:
            save_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix, volume_mean,
                            volume_std)


def solver_bSSFP():
    image_data_files, label_data_files = get_nii_files(bssfp_folder)

    image_data_files = resample_image(image_data_files)
    label_data_files = resample_image(label_data_files, is_label=True)

    image_data_files, label_data_files = crop_images_and_labels(image_data_files, label_data_files)

    extract_all_view_slices(image_data_files, label_data_files, image_type="bSSFP")


def solver_LGE():
    image_data_files, label_data_files = get_nii_files(lge_folder)

    image_data_files = resample_image(image_data_files)
    label_data_files = resample_image(label_data_files, is_label=True)

    image_data_files, label_data_files = crop_images_and_labels(image_data_files, label_data_files)

    extract_all_view_slices(image_data_files, label_data_files, image_type="LGE")


if __name__ == "__main__":
    solver_bSSFP()
    solver_LGE()

