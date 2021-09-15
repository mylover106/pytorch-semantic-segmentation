import torch
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

color_map = {
  (0, 0, 0): 0,
  (128, 0, 0): 1,  # building
  (128, 64, 128): 2,  # Road
  (0, 128, 0): 3,  # Tree
  (128, 128, 0): 4,  # vegetation
  (64, 0, 128): 5,  # Moving Car
  (192, 0, 192): 6,  # Static car
  (64, 64, 0): 7,  # Human
}


class UrbanDataset(Dataset):
  """Urban Scene Segmentation Dataset
  """

  def __init__(self, input_path):
    """
    Args:
        input_path (string): input path contain /Images folder and annotation /Labels
    """

    self.image_path = input_path + '/Images'
    self.annotation_path = input_path + '/Labels'

    self.images = glob.glob(self.image_path + '/*')
    self.annotations = glob.glob(self.annotation_path + '/*.png')

    augmt = T.Compose([
      T.ToTensor(),
      # T.RandomVerticalFlip(),
      T.RandomHorizontalFlip(),
      # T.RandomCrop((1112, 512)),
      T.RandomPerspective(interpolation=T.InterpolationMode.NEAREST)
      ])
    self.augmt = augmt
  
  def _parse_anno(self, name):
    anno = Image.open(name)
    anno = np.array(anno)
    labels = np.zeros(anno.shape[:2]).astype(np.uint8)
    for color in color_map:
      labels[(anno == color).all(axis=2)] = color_map[color]
    
    return labels
  
  def _parse_image(self, name):
    image = Image.open(name)
    image = np.array(image)
    # image = image.transpose(2, 0, 1)
    return image

  def _augment(self, image, anno):
    # stack image and anno
    anno = anno[..., np.newaxis]
    data = np.concatenate((image, anno), axis=2)
    data = self.augmt(data)
    image, anno = data[:3], data[3]
    return image, anno

  def __getitem__(self, idx):
    """return Tensor data according to idx

    Args:
        idx (int): index of data
    
    Retuns:
        image & annotation in shape [B, 3, H, W] & [B, H, W]
    """

    image_name = self.images[idx]
    anno_name = self.annotations[idx]

    image = self._parse_image(image_name)
    anno = self._parse_anno(anno_name)

    image, anno = self._augment(image, anno)
    return image, anno

  

if __name__ == '__main__':
  dataset = UrbanDataset('../train_data/uavid_train/seq1')

  for i in range(10):
    img, anno = dataset[i]
    to_image = T.ToPILImage()
    img = to_image(img)
    anno = to_image(anno)
    img, anno = np.array(img), np.array(anno)
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(anno))
    plt.show()
