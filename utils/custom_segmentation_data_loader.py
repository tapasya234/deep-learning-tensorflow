import numpy as np
import albumentations as A
import cv2
from typing import List, Tuple

from keras.utils import Sequence


class CustomSegmentationDataLoader(Sequence):
    """
    Custom class for creating training and validation segmntation dataset objects.
    """

    def __init__(
        self,
        batch_size,
        image_size,
        image_paths,
        mask_paths,
        num_classes: int,
        should_augment_dataset: bool,
        coloursList: List[Tuple],
    ):
        assert len(image_paths) == len(
            mask_paths
        ), "The lenghts of the image_paths and mask_paths vary."

        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.should_augment_dataset = should_augment_dataset

        self.x = np.empty((self.batch_size,) + self.image_size + (3,), dtype=np.float32)
        self.y = np.empty((self.batch_size,) + self.image_size, dtype=np.float32)

        if self.should_augment_dataset:
            self.train_transforms = self.transforms()

        self.resize_transforms = self.resize()

        self.assignColoursToClassIDs(coloursList)

    def __len__(self):
        """
        Returns the number of batches created for the dataset.
        """
        return len(self.image_paths) // self.batch_size

    def assignColoursToClassIDs(self, coloursList: List[Tuple]):
        """
        A dictionary mapping of classID to RGB colour used to render
        the predicted segmentation map.

        :param coloursList: List of colours that will be associated with each class.
        :type coloursList: List[Tuple]
        """
        assert self.num_classes == len(
            coloursList
        ), "Provided colours do not match number of classes"
        self.classIDToColour = {}
        for i, colour in enumerate(coloursList):
            self.classIDToColour[i] = colour

    def rgb_to_one_hot(self, rgb_image: np.ndarray):
        """
        Converts the RGB image to one-hot encoded images where the number of
        channels will be the same as the number of classes in the dataset.

        :param rgb_image: The mask image which needs to one-hot encoded.
        :type rgb_image: np.ndarray
        """
        shape = rgb_image.shape[:2] + (self.num_classes,)
        arr = np.zeros(shape, dtype=np.float32)

        for i in range(self.num_classes):
            arr[:, :, i] = np.all(
                rgb_image.reshape((-1, 3)) == self.classIDToColour[i], axis=1
            ).reshape(shape[:2])

        return arr

    def transforms(self):
        """
        Adds the augmentation transforms that will be applied on a dataset.
        Should typically be applied only on the training dataset.
        """

        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.1,
                    rotate_limit=0.2,
                    shift_limit=0.2,
                    p=0.5,
                    border_mode=0,
                ),
            ]
        )

    def resize(self):
        """
        Add the resize transforms that will be applied on a dataset.
        """

        return A.Compose(
            [
                A.Resize(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    interpolation=cv2.INTER_NEAREST,
                    p=1,
                )
            ]
        )

    def reset_batch(self):
        self.x.fill(0.0)
        self.y.fill(0.0)

    def __getitem__(self, batchIndex: int) -> Tuple:
        """
        Gets the batch associated at the provided index.

        :param batchIndex: The index at which the associated batch should be returned.
        :type batchIndex: int

        :return: Returns a tuple of list of images and the associated mask.
        The image is of shape [HEIGHT, WIDTH, 3] and the mask is of shape [HEIGHT, WIDTH].
        :rtype: Tuple
        """
        self.reset_batch()

        startIndex = batchIndex * self.batch_size
        for idx, (img_path, mask_path) in enumerate(
            zip(
                self.image_paths[startIndex : startIndex + self.batch_size],
                self.mask_paths[startIndex : startIndex + self.batch_size],
            )
        ):
            # Read the image and mask and convert to RGB format.
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # Resize the image and mask.
            resized = self.resize_transforms(image=img, mask=mask)
            img, mask = resized["image"], resized["mask"]

            # Augment the image and mask, if it should.
            if self.should_augment_dataset:
                augmented = self.train_transforms(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            # Store the normalised image in X.
            self.x[idx] = img / 255

            # Convert the RGB segmentation mask to multi-channel (one-hot encoded)
            # arrays where each channel represents a single pixel whose pixel
            # values are either 0 or 1.
            # 1 represents a pixel location associated with the class
            # that corresponds to the channel.
            mask = self.rgb_to_one_hot(mask)

            # Convert the multi-channel mask to a single channel (grayscale)
            # representation whose values contain the classIDs for each class,
            # essentially collapsing the one-hot encoded arrays into a single channel.
            self.y[idx] = mask.argmax(-1)

        return self.x, self.y
