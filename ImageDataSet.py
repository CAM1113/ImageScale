from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from DataEnhance import get_enforce_image


class ImageDataSet(Dataset):
    def __init__(self, data_path, input_shape=(512, 512)):
        file = open(data_path, "r")
        s = file.read().strip()
        file.close()
        self.lines = s.split("\n")
        self.input_shape = input_shape

    def __getitem__(self, item):
        line = self.lines[item]
        line = line.split(" ")
        image_path = line[0]
        image = Image.open(image_path).convert('RGB')
        image = get_enforce_image(image, input_shape=self.input_shape)
        return image

    def __len__(self):
        return len(self.lines)


def dataset_collate(batch):
    target_images = []
    scale_images = []
    enlarge_images = []
    for img in batch:
        img = img[0]
        w, h = img.size
        scale_image = img.resize((int(w / 2),int(h / 2)), Image.BICUBIC)
        enlarge_image = scale_image.resize((w, h), Image.BICUBIC)
        _img = np.array(img)
        _img = torch.from_numpy(_img).float().permute(dims=(2, 0, 1))
        _img = _img / 255.0
        target_images.append(_img)

        _scale_image = np.array(scale_image)
        _scale_image = torch.from_numpy(_scale_image).float().permute(dims=(2, 0, 1))
        _scale_image = _scale_image / 255.0
        scale_images.append(_scale_image)

        _enlarge_image = np.array(enlarge_image)
        _enlarge_image = torch.from_numpy(_enlarge_image).float().permute(dims=(2, 0, 1))
        _enlarge_image = _enlarge_image / 255.0
        enlarge_images.append(_enlarge_image)

    target_images = torch.stack(target_images)
    scale_images = torch.stack(scale_images)
    enlarge_images = torch.stack(enlarge_images)
    return target_images, scale_images, enlarge_images


from torch.utils.data import DataLoader


def t():
    batch_size = 5
    dataset = ImageDataSet(data_path='./train.txt', input_shape=(512, 512))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True,
                             shuffle=True, collate_fn=dataset_collate)
    for data in data_loader:
        origin_images, scale_images, enlarge_images = data
        print("origin_images.shape = {}".format(origin_images.shape))
        print("scale_images.shape = {}".format(scale_images.shape))
        print("enlarge_images.shape = {}".format(enlarge_images.shape))


if __name__ == '__main__':
    t()
