"""
    my_visualize_trajectories_on_scene.py
"""

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Disable GPU usage
import sionna
import sionna_vispy
import pandas as pd
import numpy as np
np.random.seed(42)  # You can use any integer here to set the seed
import time

import vispy.scene as vscene
from vispy.visuals import transforms

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

from utils_sionna import load_dict_from_json,  setup_antenna_arrays, add_base_stations_to_scene
from utils_trajectories import discretize_by_splines
from vispy.scene.visuals import Text
import vispy.io as io

# Screen shot part to export the 3D visualisation to an image
from vispy.gloo.util import _screenshot

def _add_text_label(canvas, position, text):
    
    # Adjust z position for better visibility
    position = list(position)
    
    # Create a TextVisual object
    text_visual = Text(text=text, color='white', font_size=15000,bold=True,pos=position,
                             anchor_x='center', anchor_y='center',name=f'BS_Label_{text}')
    canvas._add_child(text_visual, position, position, persist=True)
    
    
def user_path_to_segments(user_positions):
    """
    Converts a user path to a list of line segments.

    Input
    -----
    user_positions : [N, 3], float
        User positions in the format [x, y, z].

    Output
    ------
    starts,ends : [N-1, 3], float
        Starts and Ends of  segments.
    """
    # Convert user_positions to a NumPy array if it isn't already
    user_positions = np.array(user_positions)

    # Efficiently slice the positions to create start and end arrays
    starts = user_positions[:-1]
    ends = user_positions[1:]
    
    return starts, ends

if __name__=="__main__":
    
    # Load the main parameters of the scenario
    main_params = load_dict_from_json("./Scenario/simulation_params.json")

    
    # Path to the scene file
    xml_file_path = main_params["scene_xml_fn"]

    # Cordinnates of base stations and user deplacement in cartesian coordinates system
    projected_mouvement_path_filename = main_params["user_mouvement_path_fn"]
    
    # Cordinnates of base stations and user deplacement in cartesian cordinates system
    projected_base_station_cordinates_file_name =main_params["base_station_locations_fn"]
    
    # Load the scene
    scene = sionna.rt.load_scene(xml_file_path)

    # Load base stations cordinates into a data frame
    bs_cordinates_df = pd.read_csv(projected_base_station_cordinates_file_name)
    
    # Set up the antennas array
    setup_antenna_arrays(scene, main_params)
    
    # Add the base stations to the scene
    add_base_stations_to_scene(scene, bs_cordinates_df, main_params["base_station_height"])


    # No smoothed version of the trajectory // the output of the probabilistic roadmap
    user_path_df = pd.read_csv(projected_mouvement_path_filename)

    # Smoothed paths meta data 
    meta_data_df = pd.read_csv("./Scenario/0627_s4_straight/s4_straight_paths_random_start_goal_smoothed_meta_data.csv")

    # Filter the trajectories contining only valid points
    trajectories_indices = list(meta_data_df[(meta_data_df["ration_invalid"] == 0)]["index"])
    
    # Shuffle the indices randomly with the fixed seed
    shuffled_indices = np.random.permutation(trajectories_indices)


    # Select the indices to be visualized (24 hours of simulation)
    total_number_of_sampels = 0
    selected_indices = []

    # Iterate over the shuffled indices and accumulate distances
    for idx in shuffled_indices:
        # Get the 'original_path_distance' for the current index 
        original_path_distance = meta_data_df.loc[meta_data_df["index"] == idx, "original_path_distance"].values[0]
        # Get the number of points in the path
        total_number_of_sampels += int((original_path_distance/main_params["user_speed"]) * main_params["sampling_frequency"])
        # Append the index to the selected_indices list
        selected_indices.append(idx)
        
        # Check if the total number of samples is greater than or equal to 24 hours of simulation
        if total_number_of_sampels >= 3600 * 24 * main_params["sampling_frequency"]:
            break
    
    print("Number of trajectories to be processed :",len(selected_indices))
    print("Total number of samples :",total_number_of_sampels)
    
    
    # Filter the user path data frame to keep only the selected indices
    user_path_df = user_path_df[user_path_df["index"].isin(selected_indices)]
    
    # Smooth the user paths using cubic splines and sample them at a fixed rate 
    smoothed_user_path_dict, smoothed_orientations_dict, time_steps_dict = discretize_by_splines(user_path_df, main_params["sampling_frequency"],main_params["user_speed"])
    

    
    user_path_df["z"]=main_params["user_device_height"]
    
    N = len(selected_indices)  # Number of colors needed
    
    cmap_name = 'tab20b'  # Colormap name

    # Load the colormap and generate N colors
    cmap = plt.get_cmap(cmap_name, N)
    colors = [mcolors.to_hex(cmap(i)) for i in range(N)]
    
    # Create the dictionary mapping IDs to colors
    color_dict = dict(zip(selected_indices, colors))

    # Prepare original trajectory dictionary
    original_trajectory_dict = {index: user_path_df[user_path_df['index'] == index][['x', 'y', 'z']].values for index in selected_indices}    
    
    # Visualize the trajectories on the scene using VisPy
    with sionna_vispy.patch():                
        # canvas = scene.preview(show_orientations=False,show_devices=False,resolution=(800,600),fov=45)

            
        # axes = scene.visuals.XYZAxis(parent=canvas.scene)
        # axes.transform = transforms.STTransform(translate=(0, 0, 0),
        #                                         scale=(50, 50, 50))

        # grid = scene.visuals.GridLines(color=(1, 1, 1, 0.25),
        #                             parent=canvas.scene)
        # grid.transform = transforms.STTransform(translate=(0, 0, 0.1))




        canvas = scene.preview(show_orientations=False,
                           show_devices=False,
                           resolution=(800, 600),
                           fov=45)

        # ② vscene から XYZAxis と GridLines を生成
        axes = vscene.visuals.XYZAxis(parent=canvas.scene)
        axes.transform = transforms.STTransform(translate=(0, 0, 0),
                                                scale=(50, 50, 50))  # 軸の長さ

        grid = vscene.visuals.GridLines(color=(1, 1, 1, 0.25),  # RGBA (α=0.25)
                                        parent=canvas.scene)
        grid.transform = transforms.STTransform(translate=(0, 0, 0))

        
        # Sampling rate of the smoothed paths to avoid to plot all the points and surcharge the visualization
        SAMPLING_RATE = 50
        
        for i in selected_indices:
            user_positions = np.array(smoothed_user_path_dict[i])
            user_positions = user_positions[np.arange(0,user_positions.shape[0],SAMPLING_RATE)]
            
            starts, ends = user_path_to_segments(user_positions)
            
            if starts.shape[0]>0:
                canvas._plot_lines(np.vstack(starts),np.vstack(ends), color="lightgray", width=1.0)
                        
        # Add base station manually to the vizulaization with red color using personlized radius
        colors = np.full((bs_cordinates_df.shape[0], 3), [1, 0, 0])
        canvas._plot_points(bs_cordinates_df[["x", "y", "z"]].values,persist=True,radius=1,colors=colors)
        
        # Add text labels to the base stations 
        for i, row in bs_cordinates_df.iterrows():
            position = row[["x", "y", "z"]].values
            text = f"{i}"
            _add_text_label(canvas, position, text)        
    
        # Modify the camera position
        canvas.camera.distance = 800
        canvas.camera.elevation = 60
        canvas.camera.azimuth = 0
    
    canvas.show()
    # Code to save the screen shot
    # img = _screenshot()
    # image_filename = "canvas-trajectories_visualisation" + time.strftime("%Y%m%d-%H%M%S") + ".png"
    # io.write_png(image_filename,img) 
    canvas.app.run()
    



    

    

