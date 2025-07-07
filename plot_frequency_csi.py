import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sionna.channel.utils import subcarrier_frequencies
import sionna
from sionna.channel import cir_to_ofdm_channel
import tensorflow as tf
import pandas as pd
from math import ceil
from utils_sionna import positions_to_cm_indexs



from utils_sionna import load_dict_from_json, process_scene, add_base_stations_to_scene

def plot_amp_subcarrier_idx_with_slider(h_f, rx=0, tx=0, rx_ant_list=None, tx_ant_list=None, subcarrier_indexes=None):
    """
    Plots the amplitude of the subcarriers with a slider to control batch_sample_id.
    
    Parameters:
    - h_f: The CSI data (should be a 6D array) with this shape  :[num_samples,num_rx,num_rx_ant,num_tx,num_tx_ant,num_time_steps,fft_size].
    - rx: Index of the receiving base station.
    - tx: Index of the transmitting user.
    - rx_ant_list: Optional list of receiving antennas to plot. If None, plots all Rx antennas.
    - tx_ant_list: Optional list of transmitting antennas to plot. If None, plots all Tx antennas.
    - subcarrier_indexes: Optional list of specific subcarrier indices to plot.
    """
    num_samples, _, num_rx_ant, _, num_tx_ant, num_subcarriers = h_f.shape

    # Default to all antennas if not specified
    if rx_ant_list is None:
        rx_ant_list = list(range(num_rx_ant))
    if tx_ant_list is None:
        tx_ant_list = list(range(num_tx_ant))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2) 

    # Initial sample IDX to plot
    initial_sample_id = 0

    # Function to plot the data for a given batch_sample_id
    def plot_data(sample_idx):
        ax.clear()
        ax.set_title(f"Sample {sample_idx}, BS {tx} -> UE {rx}", fontsize=12)

        # Loop over the specified Rx and Tx antennas
        for rx_ant in rx_ant_list:
            for tx_ant in tx_ant_list:
                csi = h_f[sample_idx, rx, rx_ant, tx, tx_ant, :]
                csi_amplitudes = 20 * np.log10(np.abs(csi))
                if subcarrier_indexes is not None:
                    csi_amplitudes = csi_amplitudes[subcarrier_indexes]
                    ax.set_xticks(subcarrier_indexes)
                ax.plot(csi_amplitudes, label=f'Rx Antenna {rx_ant}, Tx Antenna {tx_ant}')

        ax.set_xlabel('Subcarrier Index', fontsize=16)
        ax.set_ylabel('Amplitude (dB)', fontsize=16)
        ax.grid(True)
        ax.legend()
        fig.canvas.draw_idle()

    # Initial plot
    plot_data(initial_sample_id)

    # Add a slider for controlling batch_sample_id
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Sample index', 0, num_samples - 1, valinit=initial_sample_id, valstep=1)

    # Update function for the slider
    def update(val):
        batch_sample_id = int(slider.val)
        plot_data(batch_sample_id)

    slider.on_changed(update)

    plt.show()
    
def plot_amp_subcarrier_idx_with_slider(hdf5_file_path, trajectory_index, frequencies, rx=0, tx_list=[0], rx_ant_list=None, tx_ant_list=None, subcarrier_indexes=None):
    """
    Plots the amplitude of the subcarriers with a slider to control batch_sample_id.
    
    Parameters:
    - hdf5_file_path: Path to the HDF5 file.
    - trajectory_index: The trajectory group name in the HDF5 file (e.g., 'traj_idx').
    - frequencies: The frequencies of the subcarriers.
    - rx: Index of the receiving base station.
    - tx: list of transmitters.
    - rx_ant_list: Optional list of receiving antennas to plot. If None, plots all Rx antennas.
    - tx_ant_list: Optional list of transmitting antennas to plot. If None, plots all Tx antennas.
    - subcarrier_indexes: Optional list of specific subcarrier indices to plot.
    """
    h5_file_handle = h5.File(hdf5_file_path, 'r')
    
    num_samples = h5_file_handle[trajectory_index]['a'].shape[0]
    num_rx_ant = h5_file_handle[trajectory_index]['a'].shape[2]
    num_tx_ant = h5_file_handle[trajectory_index]['a'].shape[4]
    num_tx = h5_file_handle[trajectory_index]['a'].shape[3]

    # Default to all antennas if not specified
    if rx_ant_list is None:
        rx_ant_list = list(range(num_rx_ant))
    if tx_ant_list is None:
        tx_ant_list = list(range(num_tx_ant))
    
    if tx_list is None:
        tx_list = list(range(num_tx))

    # Calculate the grid size dynamically
    num_plots = len(tx_list)
    num_cols = 1 if num_plots == 1 else 2  
    num_rows = ceil(num_plots / num_cols) 

    # Create the figure with a flexible grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8 * num_rows))
    plt.subplots_adjust(bottom=0.2) 
    
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten the axes to easily index them

    # Hide any unused subplots if the number of subplots is less than num_rows*num_cols
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # Initial sample IDX to plot
    initial_sample_id = 0

    # Function to plot the data for a given batch_sample_id
    def plot_data(sample_idx):
        # Clear all subplots
        for ax in axes:
            ax.clear()

        # Load the required sample data dynamically from the HDF5 file
        path_coefs = h5_file_handle[trajectory_index]['a'][sample_idx:sample_idx+1, ...]
        delays = h5_file_handle[trajectory_index]['tau'][sample_idx:sample_idx+1, ...]

        # Expand dimensions for consistency with cir_to_ofdm_channel
        path_coefs = np.expand_dims(path_coefs, axis=[-1])
        # Compute the CSI for the current sample
        csi = cir_to_ofdm_channel(frequencies, path_coefs, delays, normalize=False)

        # Squeeze the time step dimension (-2) to match the expected shape
        csi = np.squeeze(csi, axis=-2)

        # Loop over the specified Tx antennas and plot on each corresponding axis
        for idx, tx in enumerate(tx_list):
            ax = axes[idx]
            ax.set_title(f"Trajectory {trajectory_index} Sample {sample_idx} - Tx {tx}", fontsize=12)

            # Loop over the specified Rx antennas
            for rx_ant in rx_ant_list:
                for tx_ant in tx_ant_list:
                    csi_amplitudes = 20 * np.log10(np.abs(csi[0, rx, rx_ant, tx, tx_ant, :]))
                    if subcarrier_indexes is not None:
                        csi_amplitudes = csi_amplitudes[subcarrier_indexes]
                        ax.set_xticks(subcarrier_indexes)
                    ax.plot(csi_amplitudes, label=f'Rx Antenna {rx_ant}, Tx Antenna {tx_ant}')
            
            ax.set_xlabel('Subcarrier Index', fontsize=16)
            ax.set_ylabel('Amplitude (dB)', fontsize=16)
            ax.grid(True)
            ax.legend()

        fig.canvas.draw_idle()

    # Initial plot
    plot_data(initial_sample_id)

    # Add a slider for controlling batch_sample_id
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Sample index', 0, num_samples - 1, valinit=initial_sample_id, valstep=1)

    # Update function for the slider
    def update(val):
        batch_sample_id = int(slider.val)
        plot_data(batch_sample_id)

    slider.on_changed(update)

    plt.show()


def plot_amp_subcarrier_idx_with_slider_and_position_and_map(hdf5_file_path, trajectory_index, frequencies, rx=0, tx_list=[0], rx_ant_list=None, tx_ant_list=None, subcarrier_indexes=None, coverage_map=None):
    
    # Assuming coverage_map._path_gain is a tensor, we get the max values as follows:
    cm_max_values = coverage_map._path_gain.numpy().max(axis=0)
    # Mask the values that are equal to 0
    cm_max_values = np.ma.masked_equal(cm_max_values, 0)
    # Log-transform the values, masking 0 values ensures they stay invisible
    cm_max_values = 10 * np.log10(cm_max_values)

    # get tx positions in the coverage map
    tx_positions = [tx.position.numpy() for tx in coverage_map._scene.transmitters.values()]
    tx_positions = np.array(tx_positions)
    tx_positions = positions_to_cm_indexs(tx_positions, coverage_map)
    
    

    min, max = cm_max_values.min(), cm_max_values.max()
    
    
    h5_file_handle = h5.File(hdf5_file_path, 'r')
     
    num_samples = h5_file_handle[trajectory_index]['a'].shape[0]
    num_rx_ant = h5_file_handle[trajectory_index]['a'].shape[2]
    num_tx_ant = h5_file_handle[trajectory_index]['a'].shape[4]
    num_tx = h5_file_handle[trajectory_index]['a'].shape[3]

    # Default to all antennas if not specified
    if rx_ant_list is None:
        rx_ant_list = list(range(num_rx_ant))
    if tx_ant_list is None:
        tx_ant_list = list(range(num_tx_ant))
    
    if tx_list is None:
        tx_list = list(range(num_tx))
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.2)
    ax1, ax2 = axes.flatten()
    
    cbar = plt.colorbar(ax2.imshow(cm_max_values, cmap='viridis', vmin=min, vmax=max, origin='lower'), ax=ax2)

    initial_sample_id = 0

    def plot_data(sample_idx):
        ax1.clear()
        ax1.set_title(f"Trajectory {trajectory_index} Sample {sample_idx}", fontsize=12)

        # Load the required sample data dynamically from the HDF5 file
        path_coefs = h5_file_handle[trajectory_index]['a'][sample_idx:sample_idx+1, ...]
        delays = h5_file_handle[trajectory_index]['tau'][sample_idx:sample_idx+1, ...]

        # Expand dimensions for consistency with cir_to_ofdm_channel
        path_coefs = np.expand_dims(path_coefs, axis=[-1])
        # Compute the CSI for the current sample
        csi = cir_to_ofdm_channel(frequencies, path_coefs, delays, normalize=False)

        # Squeeze the time step dimension (-2) to match the expected shape
        csi = np.squeeze(csi, axis=-2)
        
        print("CSI shape ",csi.shape)

        # Loop over the specified Rx and Tx antennas
        for tx in tx_list:
            for rx_ant in rx_ant_list:
                for tx_ant in tx_ant_list:
                    csi_amplitudes = 20 * np.log10(np.abs(csi[0,rx, rx_ant, tx, tx_ant, :]))
                    if subcarrier_indexes is not None:
                        csi_amplitudes = csi_amplitudes[subcarrier_indexes]
                        ax1.set_xticks(subcarrier_indexes)
                    ax1.plot(csi_amplitudes, label=f'Rx Antenna {rx_ant}, Tx {tx} Antenna {tx_ant}')

        ax1.set_xlabel('Subcarrier Index', fontsize=16)
        ax1.set_ylabel('Amplitude (dB)', fontsize=16)
        ax1.grid(True)
        ax1.legend()
        
        position_data = h5_file_handle[trajectory_index]['positions'][sample_idx:sample_idx+1,...]
        position_index_in_cm=positions_to_cm_indexs(position_data,coverage_map)
    
        
        ax2.clear() 
        ax2.scatter(position_index_in_cm[:, 0], position_index_in_cm[:, 1], c='black', s=10, marker='o')
        for i,tx_position in enumerate(tx_positions):
            ax2.scatter(tx_position[0], tx_position[1], c='red', s=20, marker='x')
            ax2.annotate(i, (tx_position[0], tx_position[1]), textcoords="offset points", xytext=(0, 8), ha='center')
        
        ax2.imshow(cm_max_values, cmap="viridis", vmin=min, vmax=max, origin='lower')
        ax2.set_title(f"Position at Sample {sample_idx}", fontsize=12)
        ax2.set_xlabel('X Position', fontsize=16)
        ax2.set_ylabel('Y Position', fontsize=16)
        ax2.grid(True)
        
        fig.canvas.draw_idle()

    plot_data(initial_sample_id)

    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Sample index', 0, num_samples - 1, valinit=initial_sample_id, valstep=1)

    def update(val):
        batch_sample_id = int(slider.val)
        plot_data(batch_sample_id)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()  # Ensure this is present to show the figure


def get_freq_domain_CSI_from_hdf5_file(hdf5_file_path,trajectory_index,frequencies,subsampling_rate=1,normalize=False):
    
    """
    Get the CSI data from an HDF5 file.
    
    Parameters:
    - hdf5_file_path: The path to the HDF5 file.
    - trajectory_index: The trajectory group name in the h5 file : format 'traj_idx'.
    - frequencies: The frequencies of the subcarriers.
    
    Returns:
    - h_f: Frequency domain CSI of shape [num_samples,num_rx,num_rx_ant,num_tx,num_tx_ant,num_time_steps,fft_size].
    """
    with h5.File(hdf5_file_path, 'r') as f:
        
        path_coefs = f[trajectory_index]['a'][::subsampling_rate,...]
        delays = f[trajectory_index]['tau'][::subsampling_rate,...]

    path_coefs = np.expand_dims(path_coefs,axis=[-1])
    
    frequecy_csi = cir_to_ofdm_channel(frequencies,path_coefs,delays,normalize=False)
    
    # squeez the time step dimension (-2) to have the excpected shape
    frequecy_csi = np.squeeze(frequecy_csi,axis=-2)

    
    return frequecy_csi

# Plot the CSI data and the positions of the users on the coverage map on different subplots
def plot_csi_and_position_on_coverage_map(hdf5_file_path, frequencies, rx=0, tx_list=[0], rx_ant_list=None, tx_ant_list=None, subcarrier_indexes=None, coverage_map=None):
    """
    Plots the amplitude of the subcarriers with a slider to control the sample and the trajectory.
    
    Parameters:
    - hdf5_file_path: Path to the HDF5 file.
    - frequencies: The frequencies of the subcarriers.
    - rx: Index of the receiving base station.
    - tx_list: list of transmitters to consider in the plot.
    - rx_ant_list: Optional list of receiving antennas to plot. If None, plots all Rx antennas.
    - tx_ant_list: Optional list of transmitting antennas to plot. If None, plots all Tx antennas.
    - subcarrier_indexes: Optional list of specific subcarrier indices to plot.
    - coverage_map: The coverage map object containing the path gain tensor.
    """
    # Assuming coverage_map._path_gain is a tensor, we get the max values as follows:
    cm_max_values = coverage_map._path_gain.numpy().max(axis=0)
    # Mask the values that are equal to 0
    cm_max_values = np.ma.masked_equal(cm_max_values, 0)
    # Log-transform the values, masking 0 values ensures they stay invisible
    cm_max_values = 10 * np.log10(cm_max_values)
    min, max = cm_max_values.min(), cm_max_values.max()

    # get tx positions in the coverage map
    tx_positions = [tx.position.numpy() for tx in coverage_map._scene.transmitters.values()]
    tx_positions = np.array(tx_positions)
    tx_positions = positions_to_cm_indexs(tx_positions, coverage_map)

    h5_file_handle = h5.File(hdf5_file_path, 'r')

    all_trajectories = list(h5_file_handle.keys())

    trajectory_index = all_trajectories[0]

    num_samples = h5_file_handle[trajectory_index]['a'].shape[0]
    num_rx_ant = h5_file_handle[trajectory_index]['a'].shape[2]
    num_tx_ant = h5_file_handle[trajectory_index]['a'].shape[4]
    num_tx = h5_file_handle[trajectory_index]['a'].shape[3]

    # Default to all antennas if not specified
    if rx_ant_list is None:
        rx_ant_list = list(range(num_rx_ant))
    if tx_ant_list is None:
        tx_ant_list = list(range(num_tx_ant))
    
    if tx_list is None:
        tx_list = list(range(num_tx))

    # Calculate the grid size dynamically
    num_plots = len(tx_list)+1
    num_cols = 1 if num_plots == 1 else 4
    num_rows = ceil(num_plots / num_cols) 

    # Create the figure with a flexible grid layout
    fig, axes = plt.subplots(num_rows, num_cols)
    # plt.subplots_adjust(bottom=0.15,top=0.95)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.2, wspace=0.4, hspace=0.6)

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten the axes to easily index them

    # Hide any unused subplots if the number of subplots is less than num_rows*num_cols
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
        fig.delaxes(axes[i])


    # Initial sample IDX to plot
    initial_sample_id = 0

    trajdisplayname = fig.text(0.92,0.07,f"{trajectory_index}",va='top', ha = 'left')

    # Function to plot the data for a given batch_sample_id
    def plot_data(trajectory, sample_idx):
        # Clear all subplots
        for ax in axes:
            ax.clear()

        # Load the required sample data dynamically from the HDF5 file
        path_coefs = h5_file_handle[trajectory]['a'][sample_idx:sample_idx+1, ...]
        delays = h5_file_handle[trajectory]['tau'][sample_idx:sample_idx+1, ...]

        # Expand dimensions for consistency with cir_to_ofdm_channel
        path_coefs = np.expand_dims(path_coefs, axis=[-1])
        # Compute the CSI for the current sample
        csi = cir_to_ofdm_channel(frequencies, path_coefs, delays, normalize=False)

        # Squeeze the time step dimension (-2) to match the expected shape
        csi = np.squeeze(csi, axis=-2)

        position_data = h5_file_handle[trajectory]['positions']
        position_index_in_cm=positions_to_cm_indexs(position_data,coverage_map)

        ax_map = axes[0]
        ax_map.clear()

        ax_map.scatter(position_index_in_cm[:, 0], position_index_in_cm[:, 1], c='gray', s=1, marker='.')

        ax_map.scatter(position_index_in_cm[sample_idx, 0], position_index_in_cm[sample_idx, 1], c='black', s=10, marker='o')
        for i,tx_position in enumerate(tx_positions):
            ax_map.scatter(tx_position[0], tx_position[1], c='red', s=20, marker='x')
            ax_map.annotate(f"Tx {i}", (tx_position[0], tx_position[1]), textcoords="offset points", xytext=(0, 8), ha='center')

        ax_map.imshow(cm_max_values, cmap="viridis", vmin=min, vmax=max, origin='lower')
        # ax_map.grid(True)
        # axes[1].axis('off')

        # Loop over the specified Tx antennas and plot on each corresponding axis
        for idx, tx in enumerate(tx_list):
            ax = axes[idx+1]
#            ax.set_title(f"Trajectory {trajectory_index} Sample {sample_idx} - Tx {tx}", fontsize=12)
            ax.set_title(f"Tx {tx}", fontsize=10)

            # Loop over the specified Rx antennas
            for rx_ant in rx_ant_list:
                for tx_ant in tx_ant_list:
                    csi_amplitudes = 20 * np.log10(np.abs(csi[0, rx, rx_ant, tx, tx_ant, :]))
                    if subcarrier_indexes is not None:
                        csi_amplitudes = csi_amplitudes[subcarrier_indexes]
                        ax.set_xticks(subcarrier_indexes)
#                    ax.plot(csi_amplitudes, label=f'Rx Antenna {rx_ant}, Tx Antenna {tx_ant}')
                    ax.plot(csi_amplitudes)

            ax.set_xlabel('Subcarrier Index', fontsize=8)
            ax.set_ylabel('Amplitude (dB)', fontsize=8)
            ax.grid(True)
            ax.legend()

        fig.canvas.draw_idle()

    # Initial plot
#    fig.suptitle(f"Trajectory {trajectory_index}", fontsize=16)
    plot_data(trajectory_index, initial_sample_id)

    # Add a slider for controlling batch_sample_id
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(ax_slider, 'Sample index', 0, num_samples - 1, valinit=initial_sample_id, valstep=1)

    # Add a slider for controlling trajectory index
    ax_slider_trajectories = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider_traj = Slider(ax_slider_trajectories, 'Trajectory', 0, len(all_trajectories)-1, valinit=0, valstep=1)
    slider_traj.valtext.set_visible(False)

    # Update function for the slider
    def update(val):
        trajectory_index = all_trajectories[int(slider_traj.val)]
        batch_sample_id = int(slider.val)
        plot_data(trajectory_index, batch_sample_id)

    def update_traj(val):
        trajectory_index = all_trajectories[int(slider_traj.val)]
        num_samples = h5_file_handle[trajectory_index]['a'].shape[0]

        slider.valmax = num_samples-1
        slider.set_val(0)
        slider.ax.set_xlim(slider.valmin,slider.valmax)
        batch_sample_id = int(slider.val)
        plot_data(trajectory_index, batch_sample_id)

#        fig.suptitle(f"Trajectory {trajectory_index}", fontsize=16)
        trajdisplayname.set_text(f"{trajectory_index}")

    slider.on_changed(update)
    slider_traj.on_changed(update_traj)
    plt.show()

        

if __name__=='__main__':
    
    hdf5_file_path = './data/csi_data_24h_trajectories.h5'
    coverage_map_data = './data/coverage_map_data.h5'
    trajectory_index = 'traj_102'
    
    # Parameters for the CSI data
    fft_size = 256
    subcarrier_spacing = 60e3
    subsampling_rate = 5 # parameter to subsample the data to reduce the size of the data to be loaded , processed and plotted
    
    # Get the frequencies of the subcarriers using sionna function
    frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
     
    # plot_amp_subcarrier_idx_with_slider(hdf5_file_path, trajectory_index, frequencies, rx=0, tx_list=[0], rx_ant_list=[0], tx_ant_list=[0], subcarrier_indexes=None)
    
    # Get the coverage map data and create the coverage map object // to be used in the next plot where we will plot the CSI data and the positions of the users on the coverage map
    # # Read the path gain tensor
    tensor_path_gain = tf.io.parse_tensor(tf.io.read_file("./data/cm_path_gain_tensor"), out_type=tf.float32)    
    # Load the scene data and create the coverage map 
    params = load_dict_from_json("./Scenario/simulation_params.json")
    # Process the scene XML file - Set Up radio materials
    scene_ = process_scene("./Scenario/citi_last.xml")
    params["coverage_map_size"] = scene_.size[:-1].numpy()
    base_stations_locations_df = pd.read_csv(params["base_station_locations_fn"])
    add_base_stations_to_scene(scene_, base_stations_locations_df, params["base_station_height"])
    cm_centre =scene_.center  
    cm_centre = tf.tensor_scatter_nd_update(cm_centre, [[2]], [1.5])
    coverage_map = sionna.rt.CoverageMap(cm_centre,orientation=[0,0,0],size=scene_.size[:-1],cell_size=[5,5],path_gain=tensor_path_gain,scene=scene_,)
    coverage_map._path_gain = tensor_path_gain
    
    # plot_csi_and_position_on_coverage_map(hdf5_file_path,frequencies=frequencies,rx=[0],tx_list=[0],rx_ant_list=[0],tx_ant_list=[0],subcarrier_indexes=None,coverage_map=coverage_map)
    plot_csi_and_position_on_coverage_map(hdf5_file_path,  frequencies, rx=0, tx_list=range(10), rx_ant_list=[0], tx_ant_list=range(16), subcarrier_indexes=None, coverage_map=coverage_map)
    # plot_amp_subcarrier_idx_with_slider_and_position_and_map(hdf5_file_path, trajectory_index, frequencies, rx=0, tx_list=[0], rx_ant_list=[0], tx_ant_list=[0], subcarrier_indexes=None, coverage_map=coverage_map)