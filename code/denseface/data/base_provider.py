import numpy as np
import torch.utils.data as data

class ImagesDataSet(data.Dataset):
    """Dataset for images that provide some often used methods"""

    def _measure_mean_and_std(self):
        # for every channel in image
        means = []
        stds = []
        # for every channel in image(assume this is last dimension)
        for ch in range(self.images.shape[-1]):
            means.append(np.mean(self.images[:, :, :, ch]))
            stds.append(np.std(self.images[:, :, :, ch]))
        self._means = np.array(means, np.float32)
        self._stds = np.array(stds, np.float32)

    @property
    def images_means(self):
        if not hasattr(self, '_means'):
            self._measure_mean_and_std()
        return self._means

    @property
    def images_stds(self):
        if not hasattr(self, '_stds'):
            self._measure_mean_and_std()
        return self._stds

    def shuffle_images_and_labels(self, images, labels):
        rand_indexes = np.random.permutation(images.shape[0])
        shuffled_images = images[rand_indexes]
        shuffled_labels = labels[rand_indexes]
        return shuffled_images, shuffled_labels

    def normalize_images(self, images, normalization_type):
        """
        use this one in new torch version
        Args:
        images: numpy 4D array
        normalization_type: `str`, available choices:
            - divide_255
            - divide_256
            - by_chanels
        """
        if normalization_type == 'divide_255':
            images = images / 255
        elif normalization_type == 'divide_256':
            images = images / 256
        elif normalization_type == 'by_chanels':
            images = images.astype('float64')
            # for every channel in image(assume this is last dimension)
            for i in range(images.shape[-1]):
                images[:, :, :, i] = ((images[:, :, :, i] - self.images_means[i]) /
                            self.images_stds[i])
        else:
            print("[Error] Unknown type of normalization")
        return images

    def normalize_all_images_by_chanels(self, initial_images):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalize_image_by_chanel(initial_images[i])
        return new_images

    def normalize_image_by_chanel(self, image):
        new_image = np.zeros(image.shape)
        for chanel in range(image.shape[-1]):
            mean = np.mean(image[:, :, chanel])
            std = np.std(image[:, :, chanel])
            new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
        return new_image