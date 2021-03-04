from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import dynamics, plot, transforms
import torchvision.transforms as T


class CellDataset(Dataset):
    def __init__(self, data_dir, split, train=True):
        super().__init__()
        from pycocotools.coco import COCO

        self.data_dir = data_dir
        self.split = split
        self.train = train

        ann_file = os.path.join(data_dir, "annotations_trainval2017/annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]

        self._classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats))
        # results's labels convert to annotation labels
        self.ann_labels = {self.classes.index(v): k for k, v in self._classes.items()}

        checked_id_file = os.path.join(data_dir, "check_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            # self.check_dataset(checked_id_file)

    def __getitem__(self, i):
        # image
        img_id = self.ids[i]
        image = self.get_image(img_id)
        # label
        target = self.get_target(img_id) if self.train else {}
        # data augment
        image, target = self.transform(image, target)
        # to tensor
        image = T.ToTensor()(image)
        target = T.ToTensor()(target)
        return image, target

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        image = np.array(image.convert('RGB'))
        return image

    @staticmethod
    def convert_to_xyxy(box):
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box # new_box format: (xmin, ymin, xmax, ymax)

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        masks = []

        if len(anns) > 0:
            for ann in anns:
                mask = self.coco.annToMask(ann)
                # mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            masks = self.mask_convert(masks)

            # mask to flows, flows.shape: list of [4 x Ly x Lx] arrays
            flows = dynamics.labels_to_flows([masks], files=None)
            target = flows[0]
            return target
        else:
            return None

    def transform(self, img, label=None):
        # dataset argument
        # step1: reshape and normalize data
        img = transforms.reshape_and_normalize_data(img, channels=[2, 1], normalize=True)
        # step2: random rotate and resize
        if self.train and label is not None:
            img, label = transforms.random_rotate_and_resize(img, label[1:], scale_range=0.5)

        return img, label


    def mask_convert(self, masks):
        # use natural encoding rather than one-hot encoding
        # input:
        #   masks: [num_instances, w, h]
        # return:
        #   masks_c: [1, w, h]
        num_instances = len(masks)
        instance_number = np.arange(start=1, stop=num_instances+1)
        instance_number = instance_number[:, np.newaxis, np.newaxis]
        masks = np.stack(masks) * instance_number
        mask_c = np.max(masks, axis=0)

        return mask_c

if __name__ == '__main__':
    cocoDataset = CellDataset(data_dir='/home/lizhogn/Documents/Data/coco/',
                              split='val2017', train=True)
    image, target = cocoDataset[0]
    import matplotlib.pyplot as plt
    flow1 = target['flows'].numpy()
    flow2 = plot.dx_to_circ([flow1[2], flow1[3]])
    plt.imshow(flow2)
    plt.show()

