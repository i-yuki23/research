import tensorflow as tf
import numpy as np
from modules.voxel_data_augmentation import rotate_3d_random_angles

class VoxelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, num_rotations, angle_unit):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.num_rotations = num_rotations
        self.angle_unit = angle_unit

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        augmented_x = []
        augmented_y = []

        for x, y in zip(batch_x, batch_y):
            augmented_volumes = rotate_3d_random_angles(x, self.num_rotations, self.angle_unit)
            augmented_x.extend(augmented_volumes)
            augmented_y.extend([y] * len(augmented_volumes))
        return np.array(augmented_x), np.array(augmented_y)