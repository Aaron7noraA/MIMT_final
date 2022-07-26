import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transform: ")

        return '\n'.join(body)


class CLICTrain(VisionDataset):
    def __init__(self, root, transform):
        super(CLICTrain, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to CLIC-train dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)


class MSCOCO(VisionDataset):
    """`MS Coco <http://mscoco.org/dataset>`_ Dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``

        Example:
            .. code:: python
                import torchvision.datasets as dset
                import torchvision.transforms as transforms
                cap = dset.CocoCaptions(root = 'dir where images are',
                                        transform=transforms.ToTensor())
                print('Number of samples: ', len(cap))
                img, target = cap[3] # load 4th sample
                print("Image Size: ", img.size())
                print(target)
            Output: ::
                Number of samples: 82783
                Image Size: (3L, 427L, 640L)
                [u'A plane emitting smoke stream flying over a mountain.',
                u'A plane darts across a bright blue sky behind a mountain covered in snow',
                u'A plane leaves a contrail above the snowy mountain top.',
                u'A mountain that has a plane flying overheard in the distance.',
                u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, transform):
        super(MSCOCO, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to COCO dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)


class Kodak(VisionDataset):
    def __init__(self, root, transform):
        super(Kodak, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to Kodak dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)


class ValidData(VisionDataset):
    def __init__(self, root, transform):
        super(ValidData, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to Kodak dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)

class CustomData(VisionDataset):
    def __init__(self, root, transform, img_ext="*.png"):
        super(CustomData, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to test dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + img_ext))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.image_paths)


class Vimeo90K(data.Dataset):
    """Video Dataset

    Args:
        folder
        transform
    """

    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.folder = np.load(root+'folder.npy')
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        # random.seed(1), path
        # print(self.folder[index]), self.root+self.folder[index]
        path = self.root+self.folder[index] + \
            '/{}.png'.format(random.randint(0, 6))

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img
