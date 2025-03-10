import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os

def read_rosbag(bag_file):
    # Open the ROS bag file
    bag = rosbag.Bag(bag_file)
    
    # Lists to store magnetic field data and timestamps
    timestamps = []
    mag_field_x = []
    mag_field_y = []
    mag_field_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    accel_x = []
    accel_y = []
    accel_z = []
    
    # Iterate over all messages in the bag
    for topic, msg, t in bag.read_messages(topics=['/imu']):
        # Extract timestamp and magnetic field data
        timestamps.append(t.to_sec())  # Convert ROS Time to seconds
        mag_field_x.append(msg.mag_field.magnetic_field.x)
        mag_field_y.append(msg.mag_field.magnetic_field.y)
        mag_field_z.append(msg.mag_field.magnetic_field.z)
        
        # Extract gyroscope data (angular velocity)
        gyro_x.append(msg.imu.angular_velocity.x)
        gyro_y.append(msg.imu.angular_velocity.y)
        gyro_z.append(msg.imu.angular_velocity.z)
        
        # Extract accelerometer data
        accel_x.append(msg.imu.linear_acceleration.x)
        accel_y.append(msg.imu.linear_acceleration.y)
        accel_z.append(msg.imu.linear_acceleration.z)
    
    bag.close()
    return np.array(timestamps), np.array(mag_field_x), np.array(mag_field_y), np.array(mag_field_z), np.array(gyro_x), np.array(gyro_y), np.array(gyro_z), np.array(accel_x), np.array(accel_y), np.array(accel_z)

def hard_iron_calibration(mag_field_x, mag_field_y, mag_field_z):
    """
    Remove hard iron bias by calculating the mean of the magnetic field data and subtracting it.
    """
    # Compute the bias (mean values)
    bias_x = np.mean(mag_field_x)
    bias_y = np.mean(mag_field_y)
    bias_z = np.mean(mag_field_z)
    
    # Subtract the bias (hard iron correction)
    mag_field_x -= bias_x
    mag_field_y -= bias_y
    mag_field_z -= bias_z
    return mag_field_x, mag_field_y, mag_field_z, np.array([bias_x, bias_y, bias_z])

# def soft_iron_calibration(mag_field_x, mag_field_y, mag_field_z):
#     """
#     Apply soft iron calibration by fitting an ellipsoid to the data
#     and transforming the coordinates into a corrected magnetic field.
#     """
#     # Stack the magnetic field data
#     mag_data = np.vstack([mag_field_x, mag_field_y, mag_field_z]).T

#     # Calculate the covariance matrix of the data
#     cov_matrix = np.cov(mag_data.T)

#     # Perform eigendecomposition of the covariance matrix
#     eigvals, eigvecs = np.linalg.eigh(cov_matrix)

#     # Compute scaling factors for each axis
#     scale_factors = 1 / np.sqrt(eigvals)

#     # Create a transformation matrix to scale the data
#     transform_matrix = eigvecs @ np.diag(scale_factors) @ eigvecs.T

#     # Apply the transformation to correct soft iron distortion
#     corrected_mag_data = mag_data @ transform_matrix.T

#     # Return the corrected magnetic field components
#     return corrected_mag_data[:, 0], corrected_mag_data[:, 1], corrected_mag_data[:, 2], transform_matrix

def soft_iron_calibration(mag_field_x, mag_field_y, mag_field_z):
    from sklearn.decomposition import PCA
    """
    Apply soft iron calibration by fitting an ellipsoid to the data
    and transforming the coordinates into a corrected magnetic field.
    """
    # Stack the magnetic field data
    mag_data = np.vstack([mag_field_x, mag_field_y, mag_field_z]).T
    # mag_data_corrected is a Nx3 array of magnetometer readings after hard iron correction
    # Step 1: Apply PCA to the magnetometer data
    pca = PCA(n_components=3)
    pca.fit(mag_data)
     # Step 2: Transform the data into the principal component space
    mag_data_transformed = pca.transform(mag_data)
    
    # Step 3: Scale the data so that it forms a sphere (normalize to unit sphere)
    max_val = np.max(np.linalg.norm(mag_data_transformed, axis=1))
    mag_data_normalized = mag_data_transformed / max_val  # Normalize to unit sphere

    # Step 4: Reverse the PCA transformation (scale and rotate the data)
    mag_data_calibrated = pca.inverse_transform(mag_data_normalized)
    
        # Step 5: Extract the corrected magnetic field components
    corrected_mag_x = mag_data_calibrated[:, 0]
    corrected_mag_y = mag_data_calibrated[:, 1]
    corrected_mag_z = mag_data_calibrated[:, 2]
    
    return corrected_mag_x, corrected_mag_y, corrected_mag_z, pca.components_, max_val

    
    # ################################3
    # # Calculate the covariance matrix of the data
    # cov_matrix = np.cov(mag_data.T)

    # # Perform eigendecomposition of the covariance matrix
    # eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # # Compute scaling factors for each axis
    # scale_factors = 1 / np.sqrt(eigvals)

    # # Create a transformation matrix to scale the data
    # transform_matrix = eigvecs @ np.diag(scale_factors) @ eigvecs.T

    # # Apply the transformation to correct soft iron distortion
    # corrected_mag_data = mag_data @ transform_matrix.T

    # # Return the corrected magnetic field components
    # return corrected_mag_data[:, 0], corrected_mag_data[:, 1], corrected_mag_data[:, 2], transform_matrix


def fit_circle(x, y):
    """
    Fit a circle to the given data points (x, y) using least squares method.
    Returns the circle's center (a, b) and radius (R).
    """
    # Calculate the mean of the data points
    N = len(x)
    xmean, ymean = x.mean(), y.mean()
    x -= xmean
    y -= ymean
    U, S, V = np.linalg.svd(np.stack((x, y)))
    
    tt = np.linspace(0, 2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    transform = np.sqrt(2/N) * U.dot(np.diag(S))   # transformation matrix
    fit_x, fit_y = transform.dot(circle) + np.array([[xmean], [ymean]])
    return fit_x, fit_y

def plot_mag_field(timestamps, mag_field_x, mag_field_y, mag_field_x_cal, mag_field_y_cal, fit_x, fit_y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # fig, axs = plt.subplots(1, 2)
    
    # Before Calibration (Original Data)
    # axs[0].plot(mag_field_x, mag_field_y, label='Circle Magnetic Field X (N)', color='blue')
    axs[0].scatter(mag_field_x, mag_field_y, label='Magnetic Field X (N)', s=2, color='blue')
    axs[0].set_title('Magnetic Field (Before Calibration)')
    axs[0].set_xlabel('Magnetic Field X (T)')
    axs[0].set_ylabel('Magnetic Field Y (T)')
    axs[0].grid(True)
    axs[0].legend()

    # After Calibration (Corrected Data)
    # axs[1].plot(mag_field_x_cal, mag_field_y_cal, label='Magnetic Field X (N) - Calibrated', color='red')
    axs[1].scatter(mag_field_x_cal, mag_field_y_cal, label='Magnetic Field X (N) - Calibrated', s=2, color='red')
    axs[1].plot(fit_x, fit_y, '--', color='green')
    axs[1].set_title('Circle Magnetic Field (After Calibration)')
    axs[1].set_xlabel('Magnetic Field X (T)')
    axs[1].set_ylabel('Magnetic Field Y (T)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    # plt.show()
    
def plot_gyro_and_rotation(timestamps, gyro_x, gyro_y, gyro_z):
    # Calculate the rotational rate in rad/s (already given in gyro data)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(timestamps, gyro_x, label="Gyro X (rad/s)", color="blue")
    axs[0].set_title("Circle Gyroscope X-axis Rotational Rate")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Angular Velocity (rad/s)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, gyro_y, label="Gyro Y (rad/s)", color="green")
    axs[1].set_title("Circle Gyroscope Y-axis Rotational Rate")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, gyro_z, label="Gyro Z (rad/s)", color="red")
    axs[2].set_title("Circle Gyroscope Z-axis Rotational Rate")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Angular Velocity (rad/s)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    # plt.show()

def plot_total_rotation_other_method(timestamps, gyro_x, gyro_y, gyro_z):
    # Calculate the total rotation by integrating angular velocities
    dt = np.diff(timestamps)
    dt = np.insert(dt, 0, 0)  # Insert zero for the first value (no change)
    
    # Integrating angular velocity to get total rotation
    total_rotation_x = np.cumsum(gyro_x * dt)
    total_rotation_y = np.cumsum(gyro_y * dt)
    total_rotation_z = np.cumsum(gyro_z * dt)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(timestamps, total_rotation_x, label="Total Rotation X (rad)", color="blue")
    axs[0].set_title("Circle Total Rotation X-axis")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Total Rotation (rad)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, total_rotation_y, label="Total Rotation Y (rad)", color="green")
    axs[1].set_title("Circle Total Rotation Y-axis")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Total Rotation (rad)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, total_rotation_z, label="Total Rotation Z (rad)", color="red")
    axs[2].set_title("Circle Total Rotation Z-axis")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Total Rotation (rad)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    # plt.show()
    
def plot_total_rotation(timestamps, gyro_x, gyro_y, gyro_z):
    # Calculate the time differences (dt)
    dt = np.diff(timestamps)
    dt = np.insert(dt, 0, 0)  # Insert zero for the first value (no change)

    # Integrating angular velocity to get total rotation using cumtrapz
    total_rotation_x = cumtrapz(gyro_x, timestamps, initial=0)
    total_rotation_y = cumtrapz(gyro_y, timestamps, initial=0)
    total_rotation_z = cumtrapz(gyro_z, timestamps, initial=0)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot for X axis rotation
    axs[0].plot(timestamps, total_rotation_x, label="Total Rotation X (rad)", color="blue")
    axs[0].set_title("Circle Total Rotation X-axis")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Total Rotation (rad)")
    axs[0].grid(True)
    axs[0].legend()

    # Plot for Y axis rotation
    axs[1].plot(timestamps, total_rotation_y, label="Total Rotation Y (rad)", color="green")
    axs[1].set_title("Circle Total Rotation Y-axis")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Total Rotation (rad)")
    axs[1].grid(True)
    axs[1].legend()

    # Plot for Z axis rotation
    axs[2].plot(timestamps, total_rotation_z, label="Total Rotation Z (rad)", color="red")
    axs[2].set_title("Circle Total Rotation Z-axis")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Total Rotation (rad)")
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    
def plot_heading(timestamps, heading, mag_x, mag_y):
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    # Heading Plot
    # plt.figure()
    # plt.plot(timestamps, heading, label='Heading (Degrees)', color='blue')
    # plt.plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    # plt.plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    # plt.title('Heading over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Heading (Degrees)')
    # plt.grid(True)
    # plt.legend()
    
    axs[0].plot(timestamps, heading, label='Heading (Degrees)', color='blue')
    # axs[0].plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    # axs[0].plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    axs[0].set_title('Circle Heading over Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Heading (Degrees)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Magnetic X vs Time
    axs[1].plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    axs[1].set_title("Circle Magnetic Field X vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Magnetic Field X (T)")
    axs[1].grid(True)
    axs[1].legend()
    
    # Magnetic X vs Time
    axs[2].plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    axs[2].set_title("Circle Magnetic Field Y vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Magnetic Field Y (T)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    # plt.show()
    
def plot_accel_velocity_displacement(timestamps, accel_x, accel_y, accel_z):
    """
    Calculate and plot acceleration, velocity, and displacement.
    
    Parameters:
    - timestamps: Time data (array of time values)
    - accel_x, accel_y, accel_z: Acceleration data along X, Y, Z axes (arrays)
    """
    
    # Calculate the time differences (dt) between consecutive timestamps
    dt = np.diff(timestamps)
    dt = np.insert(dt, 0, 0)  # Insert zero for the first value (no change)

    # Acceleration is already provided, so we just need to integrate for velocity and displacement
    # Integrating acceleration to get velocity
    # velocity_x = cumtrapz(gyro_x, timestamps, initial=0)
    # velocity_y = np.cumsum(accel_y * dt)
    # velocity_z = np.cumsum(accel_z * dt)

    # # Integrating velocity to get displacement
    # displacement_x = np.cumsum(velocity_x * dt)
    # displacement_y = np.cumsum(velocity_y * dt)
    # displacement_z = np.cumsum(velocity_z * dt)
    
    # Use cumtrapz to integrate acceleration to get velocity along each axis
    velocity_x = cumtrapz(accel_x, timestamps, initial=0)
    velocity_y = cumtrapz(accel_y, timestamps, initial=0)
    velocity_z = cumtrapz(accel_z, timestamps, initial=0)

    # Use cumtrapz to integrate velocity to get displacement along each axis
    displacement_x = cumtrapz(velocity_x, timestamps, initial=0)
    displacement_y = cumtrapz(velocity_y, timestamps, initial=0)
    displacement_z = cumtrapz(velocity_z, timestamps, initial=0)


    # Create subplots for Acceleration, Velocity, and Displacement
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot Acceleration
    axs[0].plot(timestamps, accel_x, label="Acceleration X", color="blue")
    axs[0].set_title("Circle Acceleration X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Acceleration (m/s^2)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, accel_y, label="Acceleration Y", color="green")
    axs[1].set_title("Circle Acceleration Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Acceleration (m/s^2)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, accel_z, label="Acceleration Z", color="red")
    axs[2].set_title("Circle Acceleration Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s^2)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    # Plot Velocity
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    axs[0].plot(timestamps, velocity_x, label="Velocity X", color="blue")
    axs[0].set_title("Circle Velocity X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, velocity_y, label="Velocity Y", color="green")
    axs[1].set_title("Circle Velocity Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, velocity_z, label="Velocity Z", color="red")
    axs[2].set_title("Circle Velocity Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    # Plot Displacement
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].plot(timestamps, displacement_x, label="Displacement X", color="blue")
    axs[0].set_title("Circle Displacement X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Displacement (m)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, displacement_y, label="Displacement Y", color="green")
    axs[1].set_title("Circle Displacement Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, displacement_z, label="Displacement Z", color="red")
    axs[2].set_title("Circle Displacement Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Displacement (m)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    plt.tight_layout()
    plt.show()

def apply_calibration_to_square_data(mag_field_x, mag_field_y, mag_field_z, bias, transform_matrix):
    """
    Apply the hard iron and soft iron calibration to the square data.
    """
    # Remove hard iron bias
    mag_field_x -= bias[0]
    mag_field_y -= bias[1]
    mag_field_z -= bias[2]

    # Apply soft iron transformation
    mag_data = np.vstack([mag_field_x, mag_field_y, mag_field_z]).T
    calibrated_mag_data = mag_data @ transform_matrix.T

    # Return the calibrated magnetic field components
    return calibrated_mag_data[:, 0], calibrated_mag_data[:, 1], calibrated_mag_data[:, 2]

def plot_mag_field_square(timestamps, mag_field_x, mag_field_y, mag_field_x_cal, mag_field_y_cal, fit_x, fit_y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Before Calibration (Original Data)
    axs[0].plot(mag_field_x, mag_field_y, label='Magnetic Field X (N)', color='blue')
    axs[0].set_title('Square Magnetic Field (Before Calibration)')
    axs[0].set_xlabel('Magnetic Field X (T)')
    axs[0].set_ylabel('Magnetic Field Y (T)')
    axs[0].grid(True)
    axs[0].legend()

    # After Calibration (Corrected Data)
    axs[1].plot(mag_field_x_cal, mag_field_y_cal, label='Magnetic Field X (N) - Calibrated', color='red')
    axs[1].plot(fit_x, fit_y, '--', color='green')
    axs[1].set_title('Square Magnetic Field (After Calibration)')
    axs[1].set_xlabel('Magnetic Field X (T)')
    axs[1].set_ylabel('Magnetic Field Y (T)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    # plt.show()
    
def plot_gyro_and_rotation_square(timestamps, gyro_x, gyro_y, gyro_z):
    # Calculate the rotational rate in rad/s (already given in gyro data)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(timestamps, gyro_x, label="Gyro X (rad/s)", color="blue")
    axs[0].set_title("Square Gyroscope X-axis Rotational Rate")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Angular Velocity (rad/s)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, gyro_y, label="Gyro Y (rad/s)", color="green")
    axs[1].set_title("Square Gyroscope Y-axis Rotational Rate")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, gyro_z, label="Gyro Z (rad/s)", color="red")
    axs[2].set_title("Square Gyroscope Z-axis Rotational Rate")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Angular Velocity (rad/s)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    # plt.show()
    
def plot_total_rotation_sqaure(timestamps, gyro_x, gyro_y, gyro_z):
    # Calculate the time differences (dt)
    dt = np.diff(timestamps)
    dt = np.insert(dt, 0, 0)  # Insert zero for the first value (no change)

    # Integrating angular velocity to get total rotation using cumtrapz
    total_rotation_x = cumtrapz(gyro_x, timestamps, initial=0)
    total_rotation_y = cumtrapz(gyro_y, timestamps, initial=0)
    total_rotation_z = cumtrapz(gyro_z, timestamps, initial=0)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot for X axis rotation
    axs[0].plot(timestamps, total_rotation_x, label="Total Rotation X (rad)", color="blue")
    axs[0].set_title("Square Total Rotation X-axis")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Total Rotation (rad)")
    axs[0].grid(True)
    axs[0].legend()

    # Plot for Y axis rotation
    axs[1].plot(timestamps, total_rotation_y, label="Total Rotation Y (rad)", color="green")
    axs[1].set_title("Square Total Rotation Y-axis")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Total Rotation (rad)")
    axs[1].grid(True)
    axs[1].legend()

    # Plot for Z axis rotation
    axs[2].plot(timestamps, total_rotation_z, label="Total Rotation Z (rad)", color="red")
    axs[2].set_title("Square Total Rotation Z-axis")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Total Rotation (rad)")
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    
def plot_heading_square(timestamps, heading, mag_x, mag_y):
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    # Heading Plot
    # plt.figure()
    # plt.plot(timestamps, heading, label='Heading (Degrees)', color='blue')
    # plt.plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    # plt.plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    # plt.title('Heading over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Heading (Degrees)')
    # plt.grid(True)
    # plt.legend()
    
    axs[0].plot(timestamps, heading, label='Heading (Degrees)', color='blue')
    # axs[0].plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    # axs[0].plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    axs[0].set_title('Square Heading over Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Heading (Degrees)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Magnetic X vs Time
    axs[1].plot(timestamps, mag_x, label="Magnetic Field X (T)", color="orange")
    axs[1].set_title("Square Magnetic Field X vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Magnetic Field X (T)")
    axs[1].grid(True)
    axs[1].legend()
    
    # Magnetic X vs Time
    axs[2].plot(timestamps, mag_y, label="Magnetic Field Y (T)", color="green")
    axs[2].set_title("Circle Magnetic Field Y vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Magnetic Field Y (T)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    # plt.show()
    
def plot_accel_velocity_displacement_sqaure(timestamps, accel_x, accel_y, accel_z):
    """
    Calculate and plot acceleration, velocity, and displacement.
    
    Parameters:
    - timestamps: Time data (array of time values)
    - accel_x, accel_y, accel_z: Acceleration data along X, Y, Z axes (arrays)
    """
    
    # Calculate the time differences (dt) between consecutive timestamps
    dt = np.diff(timestamps)
    dt = np.insert(dt, 0, 0)  # Insert zero for the first value (no change)

    # Acceleration is already provided, so we just need to integrate for velocity and displacement
    # Integrating acceleration to get velocity
    # velocity_x = cumtrapz(gyro_x, timestamps, initial=0)
    # velocity_y = np.cumsum(accel_y * dt)
    # velocity_z = np.cumsum(accel_z * dt)

    # # Integrating velocity to get displacement
    # displacement_x = np.cumsum(velocity_x * dt)
    # displacement_y = np.cumsum(velocity_y * dt)
    # displacement_z = np.cumsum(velocity_z * dt)
    
    # Use cumtrapz to integrate acceleration to get velocity along each axis
    velocity_x = cumtrapz(accel_x, timestamps, initial=0)
    velocity_y = cumtrapz(accel_y, timestamps, initial=0)
    velocity_z = cumtrapz(accel_z, timestamps, initial=0)

    # Use cumtrapz to integrate velocity to get displacement along each axis
    displacement_x = cumtrapz(velocity_x, timestamps, initial=0)
    displacement_y = cumtrapz(velocity_y, timestamps, initial=0)
    displacement_z = cumtrapz(velocity_z, timestamps, initial=0)


    # Create subplots for Acceleration, Velocity, and Displacement
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot Acceleration
    axs[0].plot(timestamps, accel_x, label="Acceleration X", color="blue")
    axs[0].set_title("Square Acceleration X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Acceleration (m/s^2)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, accel_y, label="Acceleration Y", color="green")
    axs[1].set_title("Square Acceleration Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Acceleration (m/s^2)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, accel_z, label="Acceleration Z", color="red")
    axs[2].set_title("Square Acceleration Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Acceleration (m/s^2)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    # Plot Velocity
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    axs[0].plot(timestamps, velocity_x, label="Velocity X", color="blue")
    axs[0].set_title("Square Velocity X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, velocity_y, label="Velocity Y", color="green")
    axs[1].set_title("Square Velocity Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, velocity_z, label="Velocity Z", color="red")
    axs[2].set_title("Square Velocity Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    # Plot Displacement
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].plot(timestamps, displacement_x, label="Displacement X", color="blue")
    axs[0].set_title("Square Displacement X vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Displacement (m)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(timestamps, displacement_y, label="Displacement Y", color="green")
    axs[1].set_title("Square Displacement Y vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Displacement (m)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(timestamps, displacement_z, label="Displacement Z", color="red")
    axs[2].set_title("Square Displacement Z vs Time")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Displacement (m)")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.subplots_adjust(hspace=1) 

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Path to your rosbag file
    current_path = os.path.join(os.getcwd(), "../data/")
    circle_imu_data_bag_path = os.path.join(current_path, 'circle_data.bag')
    square_imu_path_bag_path = os.path.join(current_path, 'square_data.bag')

    # Step 1: Read data from the rosbag
    timestamps, mag_field_x, mag_field_y, mag_field_z, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z  = read_rosbag(circle_imu_data_bag_path)
    s_timestamps, s_mag_field_x, s_mag_field_y, s_mag_field_z, s_gyro_x, s_gyro_y, s_gyro_z, s_accel_x, s_accel_y, s_accel_z  = read_rosbag(square_imu_path_bag_path)
    
    # Step 2: Hard Iron Calibration (Bias Removal)
    mag_field_x_cal1, mag_field_y_cal1, mag_field_z_cal1, bias_circle = hard_iron_calibration(mag_field_x, mag_field_y, mag_field_z)

    # Step 3: Soft Iron Calibration (Ellipsoid Fitting)
    mag_field_x_cal2, mag_field_y_cal2, mag_field_z_cal2, pac_components, max_val = soft_iron_calibration(mag_field_x_cal1, mag_field_y_cal1, mag_field_z_cal1)
    print(f"PCA Components: {pac_components} and Max val : {max_val}")
    # Step 4: Fit the best circle to the calibrated data
    fit_x, fit_y = fit_circle(mag_field_x_cal2, mag_field_y_cal2)
    
    # Step 5: Plot the magnetic field data before and after calibration
    plot_mag_field(timestamps, mag_field_x, mag_field_y, mag_field_x_cal2, mag_field_y_cal2, fit_x, fit_y)

    # Step 6: Plot Gyroscope Data (Rotational Rates)
    plot_gyro_and_rotation(timestamps, gyro_x, gyro_y, gyro_z)

    # Step 7: Plot Total Rotation (Integrated Gyroscope Data)
    plot_total_rotation(timestamps, gyro_x, gyro_y, gyro_z)
    
    #Step 8: Calculate the Heading Angle
    heading = np.degrees(np.arctan2(mag_field_y_cal2, mag_field_x_cal2))  # Convert to degrees
    plot_heading(timestamps, heading, mag_field_x_cal2, mag_field_y_cal2)  # Plot the heading angle
    
    # Step 9: Plot the acceleration, the Velocity and Displacement
    plot_accel_velocity_displacement(timestamps, accel_x, accel_y, accel_z)

    # # Apply the same calibration to square data using circle's calibration parameters
    # mag_field_x_square_cal, mag_field_y_square_cal, mag_field_z_square_cal = apply_calibration_to_square_data(
    #     s_mag_field_x, s_mag_field_y, s_mag_field_z, bias_circle, transform_matrix_circle)
    
    # # Fit a circle to the calibrated square magnetic field data
    # fit_x_square, fit_y_square = fit_circle(mag_field_x_square_cal, mag_field_y_square_cal)
    
    # # Plot the calibrated square data and the fitted circle
    # plot_mag_field_square(s_timestamps, s_mag_field_x, s_mag_field_y, mag_field_x_square_cal, mag_field_y_square_cal, fit_x_square, fit_y_square)
    
    # # Plot the square data's Rotational Rates
    # plot_gyro_and_rotation_square(s_timestamps, s_gyro_x, s_gyro_y, s_gyro_z)
    
    # #Plot the square data's Total Rotation
    # plot_total_rotation_sqaure(s_timestamps, s_gyro_x, s_gyro_y, s_gyro_z)
    
    # #Step 8: Calculate the Heading Angle of sqaure
    # heading_square = np.degrees(np.arctan2(mag_field_y_square_cal, mag_field_x_square_cal))  # Convert to degrees
    # plot_heading_square(s_timestamps, heading_square, mag_field_x_square_cal, mag_field_y_square_cal)  # Plot the heading angle
    
    # # Step 9: Plot the acceleration, the Velocity and Displacement
    # plot_accel_velocity_displacement_sqaure(s_timestamps, s_accel_x, s_accel_y, s_accel_z)