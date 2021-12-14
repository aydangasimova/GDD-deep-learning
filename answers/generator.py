from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class MyImageGenerator(Sequence):
    def __init__(self, paths, labels, batch_size):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.classes = {l: c for c, l in enumerate(set(labels))}

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        start, stop = idx * self.batch_size, (idx + 1) * self.batch_size
        batch_x = np.array([plt.imread(p) for p in self.paths[start:stop]])
        batch_y = np.array([self.classes[l] for l in self.labels[start:stop]])
        return batch_x[:, :, :, np.newaxis], to_categorical(batch_y, len(self.classes))

    def on_epoch_end(self):
        pass
