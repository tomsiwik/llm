"""MNIST dataset loading using stdlib only (urllib + gzip + struct).

Downloads from ossci-datasets mirror, caches to ~/.cache/tribe/mnist/.
"""

import gzip
import os
import struct
import urllib.request

import mlx.core as mx
import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
CACHE_DIR = os.path.expanduser("~/.cache/tribe/mnist")

FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
}


def _download(filename):
    """Download file if not cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        url = MNIST_URL + filename
        print(f"  Downloading {url}...")
        urllib.request.urlretrieve(url, path)
    return path


def _read_images(path):
    """Read IDX image file → numpy array (N, 28, 28)."""
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_labels(path):
    """Read IDX label file → numpy array (N,)."""
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    """Load MNIST dataset.

    Returns:
        (train_images, train_labels, test_images, test_labels)
        Images: numpy (N, 28, 28) float32 normalized to [0, 1]
        Labels: numpy (N,) uint8
    """
    train_imgs = _read_images(_download(FILES['train_images'])).astype(np.float32) / 255.0
    train_labels = _read_labels(_download(FILES['train_labels']))
    test_imgs = _read_images(_download(FILES['test_images'])).astype(np.float32) / 255.0
    test_labels = _read_labels(_download(FILES['test_labels']))
    return train_imgs, train_labels, test_imgs, test_labels


def make_mnist_patterns(images, labels, digit_classes, n_per_class=100, seed=0):
    """Create (image, one_hot) pattern pairs for given digit classes.

    Args:
        images: (N, 28, 28) float32
        labels: (N,) uint8
        digit_classes: list of ints, e.g. [0, 1, 2]
        n_per_class: samples per digit class
        seed: random seed for sampling

    Returns:
        list of (mx.array(28,28,1), mx.array(10,)) pattern pairs
    """
    rng = np.random.RandomState(seed)
    patterns = []
    for digit in digit_classes:
        indices = np.where(labels == digit)[0]
        chosen = rng.choice(indices, size=min(n_per_class, len(indices)), replace=False)
        for idx in chosen:
            img = mx.array(images[idx].reshape(28, 28, 1))
            one_hot = np.zeros(10, dtype=np.float32)
            one_hot[digit] = 1.0
            patterns.append((img, mx.array(one_hot)))
    return patterns
