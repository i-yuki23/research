import numpy as np
from scipy.ndimage import rotate
import itertools

SEED_TRAIN = 38
np.random.seed(SEED_TRAIN)

def rotate_random_angle(volume, angles):
    rotated_volume = np.empty_like(volume)
    for i in range(volume.shape[-1]):  # Rotate each channel independently
        rotated_volume[..., i] = rotate(volume[..., i], angles[0], axes=(1, 2), reshape=False, mode='nearest')
        rotated_volume[..., i] = rotate(rotated_volume[..., i], angles[1], axes=(0, 2), reshape=False, mode='nearest')
        rotated_volume[..., i] = rotate(rotated_volume[..., i], angles[2], axes=(0, 1), reshape=False, mode='nearest')
    return rotated_volume

def rotate_3d_random_angles(volume, num_rotations, angle_unit):
    rotations = [volume]
    
    # Define angle ranges
    x_angles = np.arange(-180, 181, angle_unit)
    y_angles = np.arange(-180, 181, angle_unit)
    z_angles = np.arange(-90, 91, angle_unit)

    # Generate all unique angle combinations excluding (0, 0, 0)
    angle_combs = [comb for comb in itertools.product(x_angles, y_angles, z_angles) if comb != (0, 0, 0)]
    # Ensure unique random rotations
    selected_combs = np.random.choice(range(len(angle_combs)), size=num_rotations)
    for idx in selected_combs:
        angles = angle_combs[idx]
        rotated_volume = rotate_random_angle(volume, angles)
        rotations.append(rotated_volume)

    return rotations



