"""
    utils_trajectories.py
"""
import numpy as np
from scipy.interpolate import CubicSpline

def get_orientations(points):
    """
    Calculate the orientation angles (yaw, pitch, roll) for a trajectory based on the tangent vectors.
    
    Parameters:
    points (np.ndarray): Array of shape (N, 3) representing the positions in the trajectory.
    
    Returns:
    np.ndarray: Array of shape (N, 3) containing orientation angles (yaw, pitch, roll) for each point.
    """
    # Calculate the tangent vectors between successive points
    tangents = np.diff(points, axis=0)
    # Compute the yaw angles based on the tangent vectors
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    # Assuming pitch and roll are zero for simplicity
    pitch = np.zeros_like(yaw)
    roll = np.zeros_like(yaw)
    # Combine the yaw, pitch, and roll angles into a single array
    orientations = np.column_stack([yaw, pitch, roll])
    # Add last orientation to match the number of points in the trajectory
    orientations = np.vstack([orientations, [orientations[-1]]])
    
    return orientations

def get_velocity_vectors_sampling_frequency(points, sampling_frequency):
    """
    Create velocity vectors for a trajectory based on successive points.
    
    Parameters:
    points (np.ndarray): Array of shape (N, 3) representing the positions in the trajectory.
    sampling_frequency (float): Number of points per second in Hz.
    
    Returns:
    np.ndarray: Array of shape (N, 3) containing velocity vectors between successive points.
    """
    points = np.array(points)
    deltas = np.diff(points, axis=0)
    # Multiply by the sampling frequency to get the velocities in m/s
    velocities = deltas* sampling_frequency
    # Add last velocity to match the number of points in the trajectory
    velocities = np.vstack([velocities, [velocities[-1]]])
    
    return velocities

def discretize_by_splines(df, sampling_frequency,user_speed):
        
        interpolated_points_dict = {}
        interpolated_orientations_dict = {}
        time_steps_dict = {}
        for traj_index in df['index'].unique():

            # Get the trajectory points
            trajectory = df[df['index'] == traj_index]
            points = trajectory[['x', 'y', 'z']].values
            
            # Calculate distances between consecutive points in  the trajectory
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            distances  = np.insert(np.cumsum(distances), 0, 0) 
            
            cumulative_times = distances/user_speed
            
            # Compute total distance and time for the path
            total_time = cumulative_times[-1]
            
            num_points = int (total_time*sampling_frequency)
            sample_times = np.linspace(0,total_time,num_points)
            
            # Interpolate each component using CubicSpline with cumulative times as the x-values
            splines = [CubicSpline(cumulative_times, points[:, i], bc_type="natural") for i in range(3)]
            # Generate smoothed path based on sample times
            smoothed_path = np.array([spline(sample_times) for spline in splines]).T
            # Calculate orientations based on the smoothed path
            orientations = get_orientations(smoothed_path)
            
            # Store the interpolated points, orientations, and sample times
            interpolated_points_dict[traj_index] = smoothed_path
            interpolated_orientations_dict[traj_index] = orientations
            time_steps_dict[traj_index] = sample_times
            
    
        return interpolated_points_dict, interpolated_orientations_dict, time_steps_dict