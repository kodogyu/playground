import numpy as np
from scipy.spatial.transform import Rotation as R

def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Create a rotation object from the given roll, pitch, and yaw
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    
    # Convert the rotation object to a rotation matrix
    rotation_matrix = r.as_matrix()
    
    return rotation_matrix

# Example usage:
roll = 180  # Roll angle in degrees
pitch = 70  # Pitch angle in degrees
yaw = 0  # Yaw angle in degrees

rotation_matrix = rpy_to_rotation_matrix(roll, pitch, yaw)
print("Rotation Matrix:")
print(rotation_matrix)
