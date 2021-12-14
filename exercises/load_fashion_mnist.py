from tempfile import mkdtemp
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import os

FASHION_TARGET = {
    0: "tshirttop",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle_boot",
}

def _load_fashion(n_train, n_test, n_valid):
    n_sum = n_test + n_valid
    (x_1, y_1), (x_2, y_2) = fashion_mnist.load_data()
    x_train, y_train = x_1[:n_train, :, :], y_1[:n_train]
    x_test, y_test = x_2[:n_test, :, :], y_2[:n_test]
    x_valid, y_valid = x_2[n_test:n_sum, :, :], y_2[n_test:n_sum]
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
        "valid": (x_valid, y_valid),
    }

def save_fashion_mnist(n_train=10000, n_test=2000, n_valid=2000, target_dir=None):
    """Save a sample of the fashion MNIST to a temporary folder."""

    datasets = _load_fashion(n_train, n_test, n_valid)

    if target_dir is None:
        target_dir = mkdtemp()

    for set_type, data in datasets.items():

        set_dir = os.path.join(target_dir, set_type)
        os.mkdir(set_dir)
        x, y = data

        for ii in range(0, len(data[0])):

            label = FASHION_TARGET[y[ii]]

            image_dir = os.path.join(set_dir, f"{label}")
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

            image_path = os.path.join(image_dir, f"{ii}.jpg")
            image = Image.fromarray(x[ii, :, :].squeeze())
            image.save(image_path)

    return target_dir