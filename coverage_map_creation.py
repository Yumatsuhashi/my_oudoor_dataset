
import argparse
import tensorflow as tf
import vispy.scene
# Disable GPU
tf.config.set_visible_devices([], 'GPU')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import sionna_vispy
import vispy
import sionna
sionna.config.seed = 42
from vispy.gloo.util import _screenshot
import vispy.io as io
from vispy.scene.widgets import ColorBarWidget
import matplotlib.colorbar as cbar
''
from utils_sionna import process_scene,setup_antenna_arrays,add_base_stations_to_scene,load_dict_from_json,setup_aerial_view_camera

def get_global_linear_min_max_cm(coverage_map_tensor):
    # Takes as input the output of compute_coverage_map function of sionna // so CoverageMap object
    # Return to calculate the global minimum and maximum values of a coverage map tensor in linear scale
        
    min_per_cm = []
    max_per_cm = []

    # coverage_map_tensor = coverage_map.path_gain

    for i in range(coverage_map_tensor.shape[0]):
        # Get the current slice of the coverage map
        cm_slice = coverage_map_tensor[i, :, :]

        # Replace zeros with a large number so they don't affect the minimum calculation
        large_number = tf.float32.max
        non_zero_mask = tf.where(cm_slice > 0, cm_slice, large_number)

        # Calculate the minimum value greater than zero and the maximum value
        min_cm_i = tf.reduce_min(non_zero_mask).numpy()
        max_cm_i = tf.reduce_max(cm_slice, axis=None, keepdims=False).numpy()

        min_per_cm.append(min_cm_i)
        max_per_cm.append(max_cm_i)
    
    # Calculate global min and max
    global_min = min(min_per_cm)
    global_max = max(max_per_cm)
    
    return global_min, global_max

def plot_local_coverage_maps(coverage_map_global, scene, main_params, max_depth=0, db_scale=False, v_min=None, v_max=None, display_combined=True, save_combined=True, save_individual=False,display_individual=True,rendering_camera="aerial-view"):
    
    # Initialize the list to store rendered images
    rendered_images = []

    # Render images for different base stations and capture them
    for i in range(len(scene.transmitters)):
        # Render the scene and capture the image
        fig = scene.render(
            camera=rendering_camera,
            coverage_map=coverage_map_global,
            cm_tx=i,
            resolution=(655, 500),
            show_paths=False,
            show_devices=True,
            cm_db_scale=db_scale,
            cm_vmin=v_min,
            cm_vmax=v_max,cm_show_color_bar=True
        )

        # Save figure to a BytesIO object and append to the list
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = plt.imread(buf)
        rendered_images.append(img)
        plt.close(fig)  # Close the figure to free memory

    # Create a grid of subplots if save_combined is True
    if save_combined or display_combined :
        num_images = len(rendered_images)
        num_cols = 5  # Number of columns in the subplot grid
        num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        fig.suptitle(f"Combined Path loss Maps (Max Depth = {max_depth})", fontsize=20)

        axes = axes.flatten()

        # Plot each image
        for i, img in enumerate(rendered_images):
            axes[i].imshow(img)
            axes[i].set_title(f"BS {i}")
            axes[i].axis('off')
            
        # Hide any remaining empty subplots
        for j in range(len(rendered_images), len(axes)):
            axes[j].axis('off')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the combined figure with max_depth in the filename
        combined_filename = os.path.join(main_params.get("coverage_map_save_dir", "coverage_maps"),
                                         f"combined_coverage_maps_max_depth_{max_depth}.jpg")
        
        if save_combined:
            print("Saving the combined figure to:", combined_filename)
            plt.savefig(combined_filename,bbox_inches='tight', pad_inches=0)

        if display_combined:
            plt.show()  # Show the combined figure
        
        plt.close()  # Close the figure to free memory

    # Logic for saving figures separately if save_individual is True
    if save_individual or display_individual:
        for i, img in enumerate(rendered_images):
            plt.figure(figsize=(10, 8))  # Adjust figure size as needed
            plt.imshow(img)
            plt.title(f"Path loss map from point of view of the BS {i}")
            plt.axis('off')  # Hide the axes for better appearance
            
            # Save each individual figure
            individual_filename = os.path.join(main_params.get("coverage_map_save_dir", "coverage_maps"),
                                                f"coverage_map_bs_{i}_max_depth_{max_depth}.jpg")
            print(f"Saving the individual figure to: {individual_filename}")
            if save_individual:  # Reusing save_combined to check saving individual
                plt.savefig(individual_filename, bbox_inches='tight', pad_inches=0)
            
            if display_individual:
                plt.show()  # Show the individual figure
            plt.close()  # Close the figure to free memory

# Example usage:
# plot_local_coverage_maps(coverage_map_global, scene, main_params, max_depth=3,db_scale=True,v_min=global_min_db,v_max=global_max_db)


# Function to plot the global coverage map with cell segmentation  -- No need for this function from the sionna version 0.19.0 --
# Can be replaced  the function cm.show_association("metric"); in the sionna version 0.19.0 //
def plot_global_cm_cell_segmentation_map(cm, max_depth=0, main_params=None, db_scale=True,save_fig=True):
    
    # Adaptation for sionna latest verison 0.19.0
    metric = "path_gain"
    # Convert the coverage map to a TensorFlow tensor    
    global_cm_tensor = getattr(cm, metric)


    # Initialize tensors for storing valid values
    valid_cells = np.logical_and(global_cm_tensor > 0., tf.math.is_finite(global_cm_tensor))
    
    # Apply dB scale if requested
    if db_scale:
        global_cm_tensor = tf.where(valid_cells, 10. * tf.experimental.numpy.log10(global_cm_tensor), global_cm_tensor)
            
    # Get valid min and max values
    v_min = global_cm_tensor[valid_cells].numpy().min()
    v_max = global_cm_tensor[valid_cells].numpy().max()
    
    # Calculate the maximum value aggregation considering only valid cells
    max_aggregated_tensor = tf.reduce_max(
        tf.where(valid_cells, global_cm_tensor, -tf.constant(float('inf'))), 
        axis=0
    ).numpy()
    
    # Calculate the index of the base station providing the maximum gain for each cell, considering only valid cells
    max_index_tensor = tf.argmax(
        tf.where(valid_cells, global_cm_tensor, -tf.constant(float('inf'))), 
        axis=0
    ).numpy()

    # Set cells where the maximum value is -inf (invalid) to a special index (e.g., -1)
    max_index_tensor[max_aggregated_tensor == -np.inf] = -1

    # Define figure size for consistency
    fig_size = (12, 6)
    
    # Create figure and plot for the maximum value aggregation
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    # Set the global title including max_depth
    fig.suptitle(f"Global CM Cell Segmentation Map (Max Depth = {max_depth})", fontsize=20)

    # Normalize the color scale based on vmin and vmax, or use global_min and global_max if not provided
    normalizer = mcolors.Normalize(vmin=v_min, vmax=v_max)
    color_map = plt.cm.get_cmap('viridis')
    
    # Plot maximum value aggregation with normalized color mapping
    im1 = ax[0].imshow(max_aggregated_tensor, origin='lower', cmap=color_map, norm=normalizer)
    plt.colorbar(im1, ax=ax[0], label='Path gain [dB]')
    ax[0].set_xlabel('Cell index (X-axis)')
    ax[0].set_ylabel('Cell index (Y-axis)')
    ax[0].set_title('Maximum Value Aggregated Path Loss Map')

    # Create a custom color map for the segmentation map
    num_base_stations = global_cm_tensor.shape[0]
    cmap = plt.get_cmap('tab20', num_base_stations + 1)
    cmap = mcolors.ListedColormap(cmap(np.arange(num_base_stations + 1)))

    # Set custom colors for under and over values
    cmap.set_under('white')
    cmap.set_over('black')

    bounds = np.arange(num_base_stations + 2) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the segmentation map
    im2 = ax[1].imshow(max_index_tensor, origin='lower', cmap=cmap, norm=norm)
    colorbar = plt.colorbar(im2, ax=ax[1], ticks=np.arange(num_base_stations + 1), label='Base Station Index')
    colorbar.ax.set_yticklabels(list(range(num_base_stations)) + ['No Gain'])

    ax[1].set_xlabel('Cell index (X-axis)')
    ax[1].set_ylabel('Cell index (Y-axis)')
    ax[1].set_title('Association Map Based on Path Gain')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title

    # Save the figure with max_depth in the filename
    combined_filename = os.path.join(
        main_params.get("coverage_map_save_dir", "coverage_maps"),
        f"global_cm_cell_segmentation_map_max_depth_{max_depth}.jpg"
    )
    
    print("Saving the combined figure to:",combined_filename)
    plt.show()
    
    if save_fig:
        plt.savefig(combined_filename)

    
    return v_min, v_max


if __name__ == "__main__":
    # Command examples:
    # python coverage_map_creation.py --simulation_params ./simulation_params.json --save_coverage_map --coverage_map_fn ./cm_path_gain_tensor --path_gain_threshold 1e-10 --vispy_visualization
    # python coverage_map_creation.py --simulation_params ./simulation_params.json --load_coverage_map --coverage_map_fn ./cm_path_gain_tensor --path_gain_threshold 1e-10 --vispy_visualization

    # Parse the command line arguments    
    parser = argparse.ArgumentParser(description="Create coverage maps for a given scene")
    
    parser.add_argument("--simulation_params", type=str, default="./Scenario/simulation_params.json",required=False, help="Path to the the json file contining params")
    
    parser.add_argument("--save_coverage_map", action="store_true", help="Enable saving of the coverage map")
    parser.add_argument("--load_coverage_map", action="store_true", help="Enable loading of the coverage map")

    # parser.set_defaults(load_coverage_map=False)
        
    parser.add_argument("--coverage_map_fn", type=str, default="./Scenario/cm_path_gain_tensor", help="Path to the coverage map file")
    
    parser.add_argument("--path_gain_threshold", type=float, default=1e-10,required=False,help="Threshold value for truncuating the path gain values (default 1e-10)")
    
    parser.add_argument("--render_visualization", action="store_true",help="Enable vispy visualization")
    
    
    args_dict = vars(parser.parse_args())
    

    # Default behaviour is to load the coverage map if the file exists 
    if not args_dict["save_coverage_map"] and not args_dict["load_coverage_map"]:
        args_dict["load_coverage_map"] = os.path.isfile(args_dict["coverage_map_fn"])

    if args_dict['load_coverage_map']:
        if not args_dict['coverage_map_fn'] or not os.path.isfile(args_dict['coverage_map_fn']):
            parser.error(
                "--coverage_map_fn must be a valid file path when --load_coverage_map is set to True."
            )
    if args_dict['save_coverage_map']:
        if not args_dict['coverage_map_fn']:
            parser.error(
                "--coverage_map_fn must be provided when --save_coverage_map is set to True."
            )
    

    # Set compute to the inverse of load
    args_dict["compute_coverage_map"] = not(args_dict["load_coverage_map"])
        
    # Load the simulation parameters from the json file
    params = load_dict_from_json(args_dict["simulation_params"])
    
    # Process the scene XML file - Set Up radio materials
    scene = process_scene(params["scene_xml_fn"],params["scattering_coefficient"],params["xpd_coefficient"])
    # Set up scene frequency
    scene.frequency = params["scene_frequency"]
    
    # Set up scene synthetic array
    scene.synthetic_array = params["synthetic_array"]
        
    # Set up coverage map at the same size as the scene //
    params["coverage_map_size"] = scene.size[:-1].numpy()
    
    # Get base stations locations
    base_stations_locations_df = pd.read_csv(params["base_station_locations_fn"])

    # Set up the antennas arrays
    setup_antenna_arrays(scene, params)
    add_base_stations_to_scene(scene, base_stations_locations_df, params["base_station_height"])
    
    # Add the aerial view camera to the scene - This camera is used to render the coverage maps
    setup_aerial_view_camera(scene)
    
    if args_dict['compute_coverage_map']==True:
        print("Computing the coverage map")
        max_depth = params.get("coverage_map_max_depth", 0)            
        # if args_dict['compute_coverage_map']==True:
        coverage_map_global = scene.coverage_map(max_depth=max_depth, 
                                                cm_cell_size=params["coverage_map_cell_size"],
                                                num_samples=int(params["coverage_map_num_samples"]),
                                                los=params.get("coverage_map_los", True),
                                                reflection=params.get("coverage_map_reflection", True),
                                                diffraction=params.get("coverage_map_diffraction", True),
                                                scattering=params.get("coverage_map_scattering", False))
    else:
        print("Loading the coverage map")
        max_depth = 0
        tensor_path_gain = tf.io.parse_tensor(tf.io.read_file(args_dict['coverage_map_fn']), out_type=tf.float32)
        # Construct the coverage map without recomputing the path gain tensor ..//
        cm_centre = scene.center
        cm_centre = tf.tensor_scatter_nd_update(cm_centre, [[2]], [1.5])
        coverage_map_global = sionna.rt.CoverageMap(cm_centre,orientation=[0,0,0],size=scene.size[:-1],cell_size=[5,5],path_gain=tensor_path_gain,scene=scene,)
        coverage_map_global._path_gain = tensor_path_gain
    
    if args_dict['save_coverage_map']:
        # Save the coverage map path gain tensor to a file
        tensor_path_gain = coverage_map_global.path_gain
        tf.io.write_file(args_dict['coverage_map_fn'], tf.io.serialize_tensor(tensor_path_gain))
        
    if args_dict['load_coverage_map']:
        
        # Truncate the path gain values based on a threshold value            
        # v = 1e-10
        v = args_dict['path_gain_threshold']
        print("Truncuating the path gain values based on a threshold value:",v)
        tensor_path_gain = tf.where(tensor_path_gain>v,tensor_path_gain,0)
        coverage_map_global._path_gain = tensor_path_gain

    # Ge the global minimum and maximum values of the coverage map in db scale
    v_min , v_max =get_global_linear_min_max_cm(tensor_path_gain)
    
    db_v_min = 10 * np.log10(v_min)
    db_v_max = 10 * np.log10(v_max)

    
    print("Global minimum value in db scale:",db_v_min)
    print("Global maximum value in db scale:",db_v_max)
    
    clim = [format(db_v_min, ".2f"), format(db_v_max, ".2f")]
    
    # Plot the global coverage map with cell segmentation 
    # global_min,global_max = plot_global_cm_cell_segmentation_map(coverage_map_global, max_depth=max_depth, main_params=params, db_scale=True,save_fig=True)
    
    tx_range = range(len(scene.transmitters))
    
    if not args_dict['render_visualization']:
        # Showing the combined coverage maps for all base stations 
        with sionna_vispy.patch():
            canvas = scene.preview(show_orientations=False,show_devices=True,resolution=(800,600),coverage_map=coverage_map_global,cm_metric="path_gain",cm_db_scale=True)
            canvas.camera.distance = 800
            canvas.camera.elevation = 60
            canvas.camera.azimuth = 0
            canvas.show()

            # Save the image to a file from the canvas screenshot
            img = _screenshot()
            image_filename =f"canvas_cm_visualisation_combined"    
            image_path = os.path.join(params.get("coverage_map_save_dir", "coverage_maps"),image_filename)
            print("Saving the image to:",image_path)
            io.write_png(image_path,img) 
            canvas.app.run()
        
        # Showing the coverage maps for each base station
        for i in tx_range:
            
            with sionna_vispy.patch():
                
                # Change the color of the transmitter to white before the visualization
                list(scene.transmitters.values())[i].color=[1,1,1]
                
                canvas = scene.preview(show_orientations=False,show_devices=True,resolution=(800,600),coverage_map=coverage_map_global,cm_metric="path_gain",cm_db_scale=True,
                                    cm_tx=i)
                    
                canvas.camera.distance = 800
                canvas.camera.elevation = 60
                canvas.camera.azimuth = 0

                # Adding Color bar (to each canvas) (uncoment lines) (514-520)
                # # Color bar limits :
                clim = [format(db_v_min, ".2f"), format(db_v_max, ".2f")]
                
                # # Create and add the ColorBarWidget
                colorbar = ColorBarWidget(
                    cmap="viridis", orientation="right", clim=clim,
                    label="Path gain [dB]", padding=(0.1, 0.1),axis_ratio=0.1)

                canvas.central_widget.add_widget(colorbar)
                colorbar.pos = (-400, 0) # Set the position of the colorbar
                
                canvas.show()
                
                # Save the image to a file from the canvas screenshot
                img = _screenshot()
                image_filename =f"canvas_cm_visualisation_bs_{i}"    
                image_path = os.path.join(params.get("coverage_map_save_dir", "coverage_maps"),image_filename)
                print("Saving the image to:",image_path)
                io.write_png(image_path,img) 
            
                canvas.app.run()

                # Change the color of the transmitter to red after the visualization
                list(scene.transmitters.values())[i].color=[1,0,0]
    else:
        #  Using render function to plot the coverage maps for each base station instead of vispy visualization 
        print("Vispy visualization is not enabled")
       # plot_local_coverage_maps(coverage_map_global, scene, params, max_depth=max_depth, db_scale=True, v_min=db_v_min, v_max=db_v_max, display_combined=True, save_combined=False, save_individual=False,display_individual=False,rendering_camera="aerial-view")
    
    # Display the colorbar alone in vispy new scene canvas //
    canvas = vispy.scene.SceneCanvas(keys='interactive')
    canvas.size = 800, 600
    canvas.show()
    grid = canvas.central_widget.add_grid(margin=10)

    cbar_widget = vispy.scene.ColorBarWidget(label="Path Gain", clim=clim,
                                    cmap="viridis", orientation="right",
                                    border_width=1)
    grid.add_widget(cbar_widget)
    cbar_widget.border_color = "#212121"
    grid.bgcolor = "#ffffff"

    cbar_widget.label.pos = (0.6, 0.5)  # Adjust relative position (x, y)

    img = _screenshot()
    image_filename =f"canvas_color_bar_visualisation"    
    image_path = os.path.join(params.get("coverage_map_save_dir", "coverage_maps"),image_filename)
    print("Saving the image to:",image_path)
    io.write_png(image_path,img) 
    canvas.app.run()
    
    # Matplotlib color bar plot //
    cmap = "viridis"  # Colormap

    # Create a figure and axis for the color bar
    fig, ax = plt.subplots(figsize=(2,8))  # Adjust the size for a vertical color bar

    # Create a scalar mappable for the color bar
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Add the color bar to the axis
    colorbar = cbar.ColorbarBase(ax, scalar_mappable, orientation="vertical", label="Path Gain [dB]")

    # Customize the color bar appearance
    colorbar.outline.set_edgecolor("#212121")  # Border color
    colorbar.outline.set_linewidth(1)         # Border width
    colorbar.ax.set_facecolor("#ffffff")      # Background color

    # Save the color bar as an image
    plt.savefig("./figures/color_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

    # Optionally display the color bar
    plt.show()
    
    # Displa and save the combined and individual coverage maps as images using the rendering camera "aerial-view" 
    ### plot_local_coverage_maps(coverage_map_global, scene, params, max_depth=max_depth, db_scale=True, v_min=db_v_min, v_max=db_v_max, display_combined=False, save_combined=True, save_individual=True)
    
    ###### Show the maximum value per cell for the path gain metric in db scale
    # fig_cm = coverage_map_global.show("path_gain")
    ###########

    ###### It s possible to show the coverage map for a specific base station by using the cm_tx parameter of the show function 
    # fig_cm = coverage_map_global.show("path_gain", cm_tx=0) -> More inforamtions in the documentation of sionna
    ###########
    
    
    ###### Association map is not working in the current version of sionna 0.19.0 using the built-in function show_association("metric") ///    
    ## Show the association map :
    # fig_association = coverage_map_global.show_association("path_gain")
    # fig_association.set_size_inches(10, 8)  # Set figure size
    # plt.show()
    ###########
