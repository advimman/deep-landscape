import os
import os.path

import torch.utils.data as data
from PIL import Image


def _make_video_dataset(dir):
    """
    Creates a 1D list of all the frames.

    1D List Structure:
    [frame0, frame1,...frameN]

    Parameters
    ----------
        dir : string
            root directory containing frames.

    Returns
    -------
        list
            1D list described above.
    """

    framesPath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framesPath.append(os.path.join(dir, image))
    return framesPath


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            2D list described above.
    """

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')


class Video(data.Dataset):
    """
    A dataloader for loading all video frames in a folder:

        |-- frame0
        |-- frame1
        :
        :
        |-- frameN

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.
    origDim : tuple
        original dimensions of the video.
    dim : tuple
        resized dimensions of the video (for CNN).

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """

    def __init__(self, root, transform=None):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, optional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
        """

        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_video_dataset(root)

        # Get dimensions of frames
        frame = _pil_loader(framesPath[0])
        self.origDim = frame.size
        self.dim = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise (RuntimeError("Found 0 files in: " + root + "\n"))

        self.root = root
        self.framesPath = framesPath
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - I0 and I1.

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            list
                sample is [I0, I1] where I0 is the frame with index
                `index` and I1 is the next frame.
        """

        sample = []
        # Loop over for all frames corresponding to the `index`.
        for framePath in [self.framesPath[index], self.framesPath[index + 1]]:
            # Open image using pil.
            image = _pil_loader(framePath, resizeDim=self.dim)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """

        # Using `-1` so that dataloader accesses only upto
        # frames [N-1, N] and not [N, N+1] which because frame
        # N+1 doesn't exist.
        return len(self.framesPath) - 1

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
