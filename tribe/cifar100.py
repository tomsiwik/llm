"""CIFAR-100 data loading and Split CIFAR-100 task creation.

Split CIFAR-100: 100 classes split into 10 tasks of 10 classes each.
Standard continual learning benchmark (class-incremental setting).
Images: 32x32x3, 50K train / 10K test.

Data is stored as numpy arrays and converted to mx.array only at batch time
to avoid creating 100K+ individual MLX arrays during setup.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def augment_batch(images, rng=None):
    """Random crop (pad 4, crop 32x32) + horizontal flip.

    Args:
        images: numpy array (N, 32, 32, 3) float32
        rng: numpy random state (default: np.random)

    Returns:
        Augmented numpy array (N, 32, 32, 3) float32
    """
    if rng is None:
        rng = np.random
    N, H, W, C = images.shape
    # Pad 4 pixels on each side with reflect
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='reflect')
    # Random crop back to 32x32
    crops = np.empty_like(images)
    for i in range(N):
        y = rng.randint(0, 9)  # 0 to 8 (padded is 40x40)
        x = rng.randint(0, 9)
        crops[i] = padded[i, y:y+H, x:x+W, :]
    # Random horizontal flip (50%)
    flip_mask = rng.random(N) > 0.5
    crops[flip_mask] = crops[flip_mask, :, ::-1, :]
    return crops
CACHE_DIR = os.path.expanduser("~/.cache/cifar-100-python")


def _download_cifar100():
    """Download and extract CIFAR-100 if not cached."""
    if os.path.exists(os.path.join(CACHE_DIR, "train")):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    tar_path = os.path.join(CACHE_DIR, "cifar-100-python.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading CIFAR-100...")
        urllib.request.urlretrieve(CIFAR100_URL, tar_path)
    print(f"  Extracting CIFAR-100...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(os.path.dirname(CACHE_DIR))


def load_cifar100():
    """Load CIFAR-100 dataset. Returns (train_imgs, train_labels, test_imgs, test_labels).

    train_imgs: (50000, 32, 32, 3) float32 [0,1]
    train_labels: (50000,) int32 fine labels 0-99
    """
    _download_cifar100()

    with open(os.path.join(CACHE_DIR, "train"), "rb") as f:
        train = pickle.load(f, encoding="bytes")
    with open(os.path.join(CACHE_DIR, "test"), "rb") as f:
        test = pickle.load(f, encoding="bytes")

    train_imgs = train[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    train_labels = np.array(train[b"fine_labels"], dtype=np.int32)
    test_imgs = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    test_labels = np.array(test[b"fine_labels"], dtype=np.int32)

    return train_imgs, train_labels, test_imgs, test_labels


class TaskData:
    """Batch-oriented task data. Avoids creating individual mx.array per sample."""

    def __init__(self, classes, images, labels):
        self.classes = classes
        self.images = images       # numpy (N, 32, 32, 3) float32
        self.labels = labels       # numpy (N,) int32
        # Pre-compute one-hot targets
        self.one_hots = np.zeros((len(labels), 100), dtype=np.float32)
        for i, l in enumerate(labels):
            self.one_hots[i, l] = 1.0

    def __len__(self):
        return len(self.labels)

    def as_mx(self):
        """Return (X, T_one_hot) as mx.arrays."""
        import mlx.core as mx
        return mx.array(self.images), mx.array(self.one_hots)

    def as_mx_labels(self):
        """Return (X, labels_int) as mx.arrays. For cross-entropy loss."""
        import mlx.core as mx
        return mx.array(self.images), mx.array(self.labels)

    def sample(self, n, rng=None):
        """Random subsample as (X_mx, T_mx)."""
        import mlx.core as mx
        if rng is None:
            rng = np.random
        idx = rng.choice(len(self), min(n, len(self)), replace=False)
        return mx.array(self.images[idx]), mx.array(self.one_hots[idx])

    def subset(self, n, seed=0):
        """Return a new TaskData with n random samples."""
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(self), min(n, len(self)), replace=False)
        return TaskData(self.classes, self.images[idx], self.labels[idx])


def make_split_cifar100(train_imgs, train_labels, test_imgs, test_labels,
                         n_tasks=10, seed=42):
    """Create Split CIFAR-100: n_tasks tasks of (100/n_tasks) classes each.

    Returns list of (TaskData_train, TaskData_test) tuples.
    """
    rng = np.random.RandomState(seed)
    classes = list(range(100))
    rng.shuffle(classes)
    classes_per_task = 100 // n_tasks

    tasks = []
    for t in range(n_tasks):
        task_classes = sorted(classes[t * classes_per_task : (t + 1) * classes_per_task])

        train_mask = np.isin(train_labels, task_classes)
        test_mask = np.isin(test_labels, task_classes)

        train_data = TaskData(task_classes, train_imgs[train_mask], train_labels[train_mask])
        test_data = TaskData(task_classes, test_imgs[test_mask], test_labels[test_mask])

        tasks.append((train_data, test_data))

    return tasks
