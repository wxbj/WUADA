import SimpleITK as sitk
import numpy as np
import os
import shutil

data_folder = 'MM-WHS'
ct_folder = 'ct_train'
mr_folder = 'mr_train'
size = 256
voxel = 1.0
save_folder = 'MM-WHS'
view = 'coronal'
keep_labels = [205, 420, 500, 820]
label_mapping = {0: 0, 205: 1, 420: 2, 500: 3, 820: 4}


def get_nii_files(folder):
    image_data_files = []
    label_data_files = []
    for file_name in os.listdir(os.path.join(data_folder, folder)):
        if file_name.endswith('image.nii.gz'):
            image_data_files.append({file_name: sitk.ReadImage(os.path.join(data_folder, folder, file_name))})
        elif file_name.endswith('label.nii.gz'):
            label_path = os.path.join(data_folder, folder, file_name)
            label = sitk.Cast(sitk.ReadImage(label_path), sitk.sitkUInt16)
            label_data_files.append({file_name: label})
    return image_data_files, label_data_files


def resample_image(data_files, new_spacing=(voxel, voxel, voxel), is_label=False, is_ct_image=False):
    def clip_high_percentile(img_np, upper_percent=0.0):
        upper_value = np.percentile(img_np, upper_percent)
        img_np = np.clip(img_np, None, upper_value)
        return img_np

    for data_file in data_files:
        for name, image in data_file.items():
            if not is_label:
                img_np = sitk.GetArrayFromImage(image)
                if is_ct_image:
                    img_np[img_np < -1024] = -1024
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

            if is_ct_image:
                resample.SetDefaultPixelValue(-1024)
            else:
                resample.SetDefaultPixelValue(0)

            resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

            resampled_img = resample.Execute(image)
            data_file[name] = resampled_img

    return data_files


def crop_images_and_labels(image_data_files, label_data_files, crop_size=(size, size, size), is_ct_image=False):
    cropped_image_data_files = []
    cropped_label_data_files = []

    for image_entry, label_entry in zip(image_data_files, label_data_files):
        image_path, image = next(iter(image_entry.items()))
        label_path, label_img = next(iter(label_entry.items()))

        size = image.GetSize()
        pad_lower = [max((c - s) // 2, 0) for s, c in zip(size, crop_size)]
        pad_upper = [max(c - s - pl, 0) for s, c, pl in zip(size, crop_size, pad_lower)]

        if any(p > 0 for p in pad_lower + pad_upper):
            image_pad = sitk.ConstantPadImageFilter()
            image_pad.SetPadLowerBound(pad_lower)
            image_pad.SetPadUpperBound(pad_upper)
            image_pad.SetConstant(-1024 if is_ct_image else 0)
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

    def orient_image(img):
        return sitk.DICOMOrient(img, 'RAI')

    def compute_foreground_mean_std(volume, modality='ct'):
        if modality.lower() == 'ct':
            mask = (volume != -1024)
        else:
            mask = (volume != 0)

        foreground = volume[mask]
        mean = np.mean(foreground)
        std = np.std(foreground)
        return mean, std

    def get_slices_by_view(img_array):
        if view == 'axial':
            return [img_array[i, :, :] for i in range(img_array.shape[0])]
        elif view == 'coronal':
            return [img_array[:, i, :] for i in range(img_array.shape[1])]
        elif view == 'sagittal':
            return [img_array[:, :, i] for i in range(img_array.shape[2])]
        else:
            raise ValueError(f"Unsupported view: {view}")

    def save_full_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix, volume_mean,
                             volume_std):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(len(image_slices)):
            prev_img = image_slices[i - 1] if i - 1 >= 0 else image_slices[0]
            curr_img = image_slices[i]
            next_img = image_slices[i + 1] if i + 1 < len(image_slices) else image_slices[-1]

            image = np.stack([prev_img, curr_img, next_img], axis=-1)

            label = label_slices[i]

            ch = (image - volume_mean) / volume_std

            np.save(os.path.join(image_dir, f"{image_prefix}_{view}_{i:03d}.npy"),
                    ch.astype(np.float32))
            np.save(os.path.join(label_dir, f"{label_prefix}_{view}_{i:03d}.npy"),
                    label.astype(np.uint8))

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

            if any(np.all((image[:, :, ch] == -1024) | (image[:, :, ch] == 0))
                   for ch in range(image.shape[-1])):
                continue

            label = label_slices[i]

            ch = (image - volume_mean) / volume_std

            np.save(os.path.join(image_dir, f"{image_prefix}_{view}_{save_idx:03d}.npy"),
                    ch.astype(np.float32))
            np.save(os.path.join(label_dir, f"{label_prefix}_{view}_{save_idx:03d}.npy"),
                    label.astype(np.uint8))

            save_idx += 1

    total = len(label_data_files)
    train_end = int(total * 0.8)

    for idx, (image_entry, label_entry) in enumerate(zip(image_data_files, label_data_files)):
        image_path, image = next(iter(image_entry.items()))
        label_path, label_img = next(iter(label_entry.items()))

        if idx < train_end:
            phase = 'train'
        else:
            phase = 'test'

        base_dir = os.path.join('..', 'datasets', save_folder, image_type, phase)
        label_prefix = os.path.basename(label_path)[:-13]
        image_prefix = os.path.basename(image_path)[:-13]
        label_dir = os.path.join(base_dir, 'labels', label_prefix)
        image_dir = os.path.join(base_dir, 'images', image_prefix)

        clear_directory(label_dir)
        clear_directory(image_dir)

        oriented_label_array = sitk.GetArrayFromImage(orient_image(label_img))
        oriented_image_array = sitk.GetArrayFromImage(orient_image(image))

        volume_mean, volume_std = compute_foreground_mean_std(oriented_image_array, modality='ct')

        label_slices = get_slices_by_view(oriented_label_array.copy())
        image_slices = get_slices_by_view(oriented_image_array.copy())

        for i in range(len(label_slices)):
            label = label_slices[i]
            new_label = np.zeros_like(label)
            for src_val, dst_val in label_mapping.items():
                new_label[label == src_val] = dst_val
            label_slices[i] = new_label

        if phase == 'test':
            save_full_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix,
                                 volume_mean,
                                 volume_std)
        else:
            save_RGB_slices(image_slices, label_slices, image_dir, label_dir, label_prefix, image_prefix, volume_mean,
                            volume_std)


def solver_ct():
    image_data_files, label_data_files = get_nii_files(ct_folder)

    image_data_files = resample_image(image_data_files, is_ct_image=True)
    label_data_files = resample_image(label_data_files, is_label=True)

    image_data_files, label_data_files = crop_images_and_labels(image_data_files, label_data_files, is_ct_image=True)

    extract_all_view_slices(image_data_files, label_data_files, image_type="ct")


def solver_mr():
    image_data_files, label_data_files = get_nii_files(mr_folder)

    image_data_files = resample_image(image_data_files)
    label_data_files = resample_image(label_data_files, is_label=True)

    image_data_files, label_data_files = crop_images_and_labels(image_data_files, label_data_files)

    extract_all_view_slices(image_data_files, label_data_files, image_type="mr")


if __name__ == "__main__":
    solver_ct()
    solver_mr()
