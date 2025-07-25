import os
import torch
import albumentations as A
import shutil
import numpy as np
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        sample = dict(image=image, mask=mask)

        if self.transform is not None:
            sample = self.transform(**sample)
            sample["mask"] = sample["mask"].unsqueeze(0)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    # implement the load dataset function here

    # check if the dataset folder has images and annotations
    images_path = os.path.join(data_path, "images")
    annotations_path = os.path.join(data_path, "annotations")
    if not os.path.exists(images_path) or not os.path.exists(annotations_path):
        OxfordPetDataset.download(data_path)

    # 將影像資料轉換成 tensor，並統一尺寸，訓練集增加多種轉換
    if mode == "train":
        transform = A.Compose([
            A.Resize(512, 512),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-30, 30),
                p=0.5
            ),
            A.Normalize(mean=(0.481, 0.449, 0.396), std=(0.269, 0.265, 0.273)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
    else:
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.481, 0.449, 0.396), std=(0.269, 0.265, 0.273)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
    
    dataset = OxfordPetDataset(data_path, mode=mode, transform=transform)
    return dataset

def calculate_mean_and_std(data_path):
    # implement the calculate mean and std function here
    images_path = os.path.join(data_path, "images")
    means = np.zeros(3)
    stds = np.zeros(3)
    total_pixels = 0
    images = []
    progess_bar = tqdm(os.listdir(images_path))
    for filename in progess_bar:
        image_path = os.path.join(images_path, filename)
        # 確認是否為圖片檔
        if not image_path.endswith(".jpg"):
            continue
        image = np.array(Image.open(image_path).convert("RGB"))
        images.append(image)
        total_pixels += np.prod(image.shape[:2])
        means += np.mean(image, axis=(0, 1))
    means /= len(images) 
    progess_bar.set_description("Calculating stds")
    for image in images:
        stds += np.sum((image - means) ** 2, axis=(0, 1))
    stds = np.sqrt(stds / total_pixels)
    means = means / 255
    stds = stds / 255
    return means, stds

if __name__ == "__main__":
    data_path = "../dataset/oxford-iiit-pet"
    means, stds = calculate_mean_and_std(data_path)
    print(f"Means: {means}, Stds: {stds}")