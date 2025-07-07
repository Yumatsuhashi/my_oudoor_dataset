
"""
    utils_sionna.py

"""

from sionna.rt import load_scene,Camera,PlanarArray, Transmitter
import json
import numpy as np
import tensorflow as tf
from sionna.rt.utils import rotation_matrix




# Function to set up an aerial view camera centered at the scene
def setup_aerial_view_camera(scene, altitude=1000):
    """
    Sets up an aerial view camera centered at the scene.

    Parameters:
    scene (Scene): The scene object.
    altitude (int): Altitude of the aerial view camera.

    Returns:
    Camera: The aerial view camera object.
    """
    if "aerial-view" in scene.cameras:
        scene.remove("aerial-view")

    scene_center = scene.center
    camera_position = [scene_center[0], scene_center[1], scene_center[2] + altitude]
    camera_look_at = [scene_center[0], scene_center[1], scene_center[2]]

    camera = Camera("aerial-view", position=camera_position, look_at=camera_look_at)
    scene.add(camera)


# Function to set up antenna arrays to the scene
def setup_antenna_arrays(scene, params):
    """
    Sets up the antenna arrays for the transmitters and receivers in the scene.

    Parameters:
    scene (Scene): The scene object.
    params (dict): Dictionary of parameters for the antenna arrays.
    """
    base_station_antenna_array = PlanarArray(
        num_rows=params["base_station_antenna_rows"],
        num_cols=params["base_station_antenna_cols"],
        vertical_spacing=params["base_station_antenna_vertical_spacing"],
        horizontal_spacing=params["base_station_antenna_horizontal_spacing"],
        pattern=params["base_station_antenna_pattern"],
        polarization=params["base_station_antenna_polarization"],
        polarization_model=params["base_station_antenna_polarization_model"]
        
    )

    user_antenna_array = PlanarArray(
        num_rows=params["user_antenna_rows"],
        num_cols=params["user_antenna_cols"],
        vertical_spacing=params["user_antenna_vertical_spacing"],
        horizontal_spacing=params["user_antenna_horizontal_spacing"],
        pattern=params["user_antenna_pattern"],
        polarization=params["user_antenna_polarization"],
        polarization_model=params["user_antenna_polarization_model"]
    )
    scene.tx_array = base_station_antenna_array
    scene.rx_array = user_antenna_array

def process_scene(xml_file_path,scattering_coefficient=0.0,xpd_coefficient=0.0):
    # Load the scene
    scene = load_scene(xml_file_path)
    print("Scene loaded successfully.")
    
    # Setting up the ground material if it does not exist or is not set
    ground_plane = scene.get("Plane")
    
    if ground_plane is None:
        print("Ground plane not found in the scene.")
    else:
        if ground_plane.radio_material is None:
            ground_material = scene.radio_materials.get("itu_medium_dry_ground")
            ground_plane.radio_material = ground_material
            
    print("Ground material assigned to the plane :",ground_plane.radio_material.name)
    
    # Important set up of radio materials properties
    present_radio_materials = []
    for object in scene.objects.keys():
        present_radio_materials.append(scene.get(object).radio_material.name)
        
    print("Used radio  materials propreties in the scen",set(present_radio_materials))
    
    for material in set(present_radio_materials):
        print("Material ",material)
        print("Use count ",scene.radio_materials.get(material).use_counter)
        scene.radio_materials.get(material).scattering_coefficient = scattering_coefficient
        scene.radio_materials.get(material).xpd_coefficient = xpd_coefficient
    
    # Get scene size and center
    scene_size = scene.size.numpy()
    scene_center = scene.center.numpy()
    print("Scene size:", scene_size)
    print("Scene center:", scene_center)
    
    return scene
    

# Save the dictionary to a JSON file
def save_dict_to_json(dictionary, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2)

# Load the dictionary from a JSON file
def load_dict_from_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def add_base_stations_to_scene(scene, base_stations_df, base_station_height, look_at_position=[0, 0, 0]):
    """
    Adds base stations to the scene.

    Parameters:
    scene (Scene): The scene object.
    base_stations_df (DataFrame): DataFrame containing base station coordinates.
    base_station_height (float): Height of the base station.
    look_at_position (list): Default look-at position if orientation is not set.
    """
    # Check if orientation columns ('yaw', 'pitch', 'roll') are present
    set_up_orientation = all(col in base_stations_df.columns for col in ["yaw", "pitch", "roll"])

    for index, row in base_stations_df.iterrows():
        tx_params = {
            "name": f"base_station_{int(row['index'])}",
            "position": [row['x'], row['y'], base_station_height],
            "color": [1.0, 0.0, 0.0]  # Red color for base stations
        }
        
        # Set 'orientation' if columns are present, otherwise set 'look_at'
        if set_up_orientation:
            tx_params["orientation"] = [row['yaw'], row['pitch'], row['roll']]
        else:
            tx_params["look_at"] = look_at_position
        
        # Create and add the Transmitter to the scene
        tx = Transmitter(**tx_params)
        scene.add(tx)
        
    print(f"Number of transmitters added: {len(scene.transmitters)}")

def remove_transmitters(scene):
    # Remove all transmitters
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)

def remove_receivers(scene):
    # Remove all receivers
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)
        
def remove_cameras(scene):
    # Remove all cameras
    for cam_name in list(scene.cameras.keys()):
        scene.remove(cam_name)

def remove_all(scene):
    remove_transmitters(scene)
    remove_receivers(scene)
    remove_cameras(scene)  



def positions_to_cm_indexs(positions_data,coverage_map):
    """
    Function to convert positions to coverage map indexs
    positions_data: [num_positions, 3] tensor
    coverage_map: sionna.rt.CoverageMap object
    
    return: [num_positions, 2] tensor : the indexs of the positions in the coverage map over the x and y axis 
    """
    
    # Get the coverage map center and expand the dimensions
    centre_expanded = tf.expand_dims(coverage_map._center, axis=0)
    
    positions_data = positions_data - centre_expanded
    
    rot_cm_2_gcs = rotation_matrix(coverage_map._orientation)
    rot_gcs_2_cm = tf.transpose(rot_cm_2_gcs)
    rot_gcs_2_cm_ = tf.expand_dims(rot_gcs_2_cm, axis=0)

    # [num_tx/num_rx/num_ris, 3]
    traj_positions = tf.linalg.matvec(rot_gcs_2_cm_, positions_data)
    
    # Keep only x and y
    # [num_tx/num_rx/num_ris, 2]
    traj_positions = traj_positions[:, :2]

    # Quantizing, using the bottom left corner as origin
    # [num_positions, 2]
    traj_positions = coverage_map._pos_to_idx_cell(traj_positions)

    return traj_positions
    