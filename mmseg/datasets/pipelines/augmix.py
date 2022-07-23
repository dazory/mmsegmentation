import copy
import inspect
import math
import warnings
import numpy as np

# import cv2
# import torch

from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from PIL import Image, ImageOps, ImageEnhance

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _, __):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, __):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level, img_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level, img_size):
  level = int_parameter(sample_level(level), img_size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(img_size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)




"""
Augmix with pillow
"""
@PIPELINES.register_module()
class AugMix:
    def __init__(self, mean, std, aug_list='augmentations', to_rgb=True, no_jsd=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

        self.mixture_width = 3
        self.mixture_depth = -1

        self.aug_prob_coeff = 1.
        self.aug_severity = 1

        self.no_jsd = no_jsd

        augmentations = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]
        augmentations_all = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y, color, contrast, brightness, sharpness
        ]
        augmentations_without_obj_translation = [
            autocontrast, equalize, posterize, solarize,
            color, contrast, brightness, sharpness
        ]
        if (aug_list == 'augmentations_without_obj_translation') or (aug_list == 'wotrans'):
            self.aug_list = augmentations_without_obj_translation
        elif aug_list == 'augmentations':
            self.aug_list = augmentations
        elif (aug_list == 'augmentations_all') or (aug_list == 'all'):
            self.aug_list = augmentations_all
        elif aug_list == 'copy':
            self.aug_list = aug_list
        else: # default = 'augmentations'
            self.aug_list = augmentations


    def __call__(self, results):

        if self.no_jsd:
            img = results['img'].copy()
            return self.aug(img)
        elif self.aug_list == 'copy':
            img = results['img'].copy()
            results['img2'] = img.copy()
            results['img3'] = img.copy()
            results['img_fields'] = ['img', 'img2', 'img3']
            return results
        else:
            img = results['img'].copy()
            results['img2'] = self.aug(img)
            results['img3'] = self.aug(img)
            results['img_fields'] = ['img', 'img2', 'img3']

            # ''' Save the result '''
            # img_orig = Image.fromarray(results['img'])
            # img_orig.save('/ws/external/data/augmix_orig.png')
            # # img_augmix1 = torch.tensor(results['img2'].clone().detach() * 255, dtype=torch.uint8).numpy()
            # img_augmix1 = results['img2']
            # img_augmix1 = torch.tensor(img_augmix1, dtype=torch.uint8)
            # img_augmix1 = img_augmix1.numpy()
            # img_augmix1= Image.fromarray(img_augmix1)
            # img_augmix1.save('/ws/external/data/augmix_1.png')
            #
            # img_augmix2 = results['img3']
            # img_augmix2 = torch.tensor(img_augmix2, dtype=torch.uint8)
            # img_augmix2 = img_augmix2.numpy()
            # img_augmix2 = Image.fromarray(img_augmix2)
            # img_augmix2.save('/ws/external/data/augmix_2.png')

            return results
        # return self.aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

    def aug(self, img):
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
        IMAGE_HEIGHT, IMAGE_WIDTH, _ = img.shape
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

        # image_aug = img.copy()
        mix = np.zeros_like(img.copy(), dtype=np.float32)
        for i in range(self.mixture_width):
            image_aug = Image.fromarray(img.copy(), "RGB")
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity, img_size)
            # Preprocessing commutes since all coefficients are convex
            image_aug = np.asarray(image_aug, dtype=np.float32)
            mix += ws[i] * image_aug
        mixed = (1 - m) * img + m * mix
        return mixed
