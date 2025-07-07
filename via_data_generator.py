import os
import math
import numpy as np
import h5py
import pandas as pd
import csv
import tensorflow as tf

USE_GPU = False

if not USE_GPU:
    # To force the use of CPU - Uncomment the following line if you want to use the GPU
    tf.config.set_visible_devices([], 'GPU')
else:
    gpus = tf.config.list_physical_devices('GPU')
    print('Number of GPUs available :', len(gpus))
    if gpus:
        gpu_num = 0 # Index of the GPU to be used (only one GPU is used in this case)
        try:
            tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        except RuntimeError as e:
            print(e)

import sionna
sionna.config.seed = 42
np.random.seed(42)  

from sionna.rt import Receiver
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
from tqdm import tqdm
import time
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs except errors

from utils_trajectories import discretize_by_splines,get_velocity_vectors_sampling_frequency
from utils_sionna import load_dict_from_json,process_scene, setup_antenna_arrays, add_base_stations_to_scene, remove_receivers

class DataGenerator:
    def __init__(self, scene, params, trajectories_indices=None,verbose=False,get_h_freq=False):
        # Contains the sionna scene instance to be used for the simulation
        self.scene = scene
        # Contains the main params of the simulation
        self.params = params
        # Contains indices of trajectories to be run 
        self.trajectories_indices = trajectories_indices or []
        # Will contain the discritized trajectories locations 
        self.user_trajectories_dict = None
        # Will contain the discritized trajectories orientations 
        self.user_orientations_dict = None
        # Will contain the time stamps of different trajectories
        self.time_steps_dict = None
        self.verbose = verbose
        
        # Cached attributes for effective data wrting to the HDF5 file 
        self.cached_datasets = {}
        self.hf = None
        
        # Chunk size for each data set to configure the HDF5 file chunk size
        self.chunk_size_per_data_set = None
        self.chunk_sizes_per_data_set_bytes = None
        self.data_types = None
        
        # Set the chunk size for each data set // while assuming synthetic array is True
        self.chunk_size_per_data_set,self.chunk_sizes_per_data_set_bytes,self.data_types = self.get_chunk_size_per_data_set(get_h_freq=get_h_freq)
        
        # Execution time dictionary
        self.execution_time = {}
        
    # Create a dictionary to store the chunk size for each data set
    def get_chunk_size_per_data_set(self,get_h_freq=False):
        nb_rx       = 1                                 # Receiver は 1 と仮定
        nb_tx       = len(self.scene.transmitters)
        ant_rx      = self.scene.rx_array.num_ant
        ant_tx      = self.scene.tx_array.num_ant

        RxTot       = nb_rx * ant_rx                    # 受信アンテナ総数
        TxTot       = nb_tx * ant_tx                    # 送信アンテナ総数
        num_rb = math.ceil(self.params["fft_size"] / 24)  # RB 数 (12SC=1RB)

        chunk_size_per_data_set = {
            "timestamps": (1,),                         # 1 タイムスタンプ/step
            "csi": (1, num_rb, TxTot, RxTot, 2)         # [time,RB,TxTot,RxTot,2]
        }

        data_types = {
            "timestamps": np.float32,
            "csi": np.float32                           # 実部/虚部分離済み
        }

        chunk_sizes_per_data_set_bytes = {
            k: np.prod(v) * np.dtype(data_types[k]).itemsize
            for k, v in chunk_size_per_data_set.items()
        }

        return chunk_size_per_data_set, chunk_sizes_per_data_set_bytes, data_types
    
    def get_max_size_per_data_set(self,num_positions):
        max_size_per_data_set = {}
        for key, value in self.chunk_size_per_data_set.items():
            if len(value) == 1:
                max_size_per_data_set[key] = (num_positions,)
            else:
                max_size_per_data_set[key] = (num_positions,) + value[1:]
            
        return max_size_per_data_set
        
    def load_user_trajectories(self):
        # Load user paths from the CSV file mentioned in the JSON configuration file
        user_path_df = pd.read_csv(self.params["user_mouvement_path_fn"])
        
        # Select all tajectories if no trajectories indices are provided
        if not self.trajectories_indices:
            self.trajectories_indices = user_path_df['index'].unique().tolist()
        else:
            # Filter the user paths based on the provided indices
            user_path_df = user_path_df[user_path_df['index'].isin(self.trajectories_indices)]
            
        # Set the user height
        user_path_df["z"]=self.params["user_device_height"]
        
        # Get params of the discritization step
        sampling_frequency = self.params["sampling_frequency"]
        user_speed = self.params["user_speed"]
        
        # Discretize the trajectory based on the method and the resolution
        discretized_trajectory_dict, discretized_orientations_dict,time_steps_dict = discretize_by_splines(user_path_df, sampling_frequency, user_speed)
        
        velocities_dict = {}
        
        # No more need for the orientation of the user and it's velocity since we are using no doppler effect
        for k in discretized_trajectory_dict.keys():  
            velocities_dict[k] = get_velocity_vectors_sampling_frequency(discretized_trajectory_dict[k],sampling_frequency)

        # Save the discretized paths and orientations as attributes 
        # Format will be key : np.array of positions/orientations  for each key [N,3] and [N,1] for time steps 
        self.user_trajectories_dict = discretized_trajectory_dict
        self.user_orientations_dict  = discretized_orientations_dict
        self.time_steps_dict = time_steps_dict
        self.user_velocities_dict = velocities_dict
        
    def add_users_to_scene(self,traj_index,start_point_index=0,n_user_in_parallel=1):

        for i in range(min(n_user_in_parallel,len(self.user_trajectories_dict[traj_index]))):
            self.scene.add(Receiver(name=f"user_{i}",
                                    position=self.user_trajectories_dict[traj_index][start_point_index+i],
                                    orientation=self.user_orientations_dict[traj_index][start_point_index+i],
                                    color=[0.0, 0.0, 0.0]))

    def update_users_in_scene(self,traj_index,start_point_index=0,n_user_in_parallel=1):
        
            new_n_user_in_parallel = n_user_in_parallel
            
            # Delete the users that are not needed for the current itteration
            if start_point_index+n_user_in_parallel >= len(self.user_trajectories_dict[traj_index]):
                new_n_user_in_parallel = len(self.user_trajectories_dict[traj_index]) - start_point_index
                                
                # Supress the users that are not needed for the current itteration
                for s in range(new_n_user_in_parallel,n_user_in_parallel):
                    self.scene.remove(f"user_{s}")
                    
            # Update the position of the users
            for i in range(new_n_user_in_parallel):
                self.scene.get(f"user_{i}").position = self.user_trajectories_dict[traj_index][start_point_index+i]
                self.scene.get(f"user_{i}").orientation = self.user_orientations_dict[traj_index][start_point_index+i]

        
    def compute_scene_paths(self):
        return self.scene.compute_paths(
            max_depth=self.params["max_depth"],
            method=self.params["method"],
            num_samples=self.params["num_samples"],
            los=self.params["los"],
            reflection=self.params["reflection"],
            diffraction=self.params["diffraction"],
            scattering=self.params["scattering"],
            ris=self.params["ris"],
            scat_keep_prob=self.params["scat_keep_prob"],
            edge_diffraction=self.params["edge_diffraction"],
            check_scene=self.params["check_scene"],
            scat_random_phases=self.params["scat_random_phases"]
        )

    def generate_cir_per_batch(self,trajectories_indices=None, batch_size=10,n_positions_in_parallel=1,file_name=None,get_h_freq=True):
        '''Generate OFDM MIMO channel impulse response (CIR) data with Doppler effect for each user trajectory.
        Parameters:
        - trajectories_indices (list): List of indices of the trajectories to be processed. If None, trajectories indices set in the scene_manager object will be used to select the trajectories to be processed. 
        - batch_size (int): Number of points to process in each batch -> Each batch will be processed and saved to the HDF5 file until the end of the trajectory total points
        - n_positions_in_parallel (int): Number of users to be processed in parallel
        - file_name (str): Name of the HDF5 file to save the data. If None, the file name will be generated based on the trajectories indices.
        - get_h_freq (bool): If True, the frequency domain CSI will be computed and saved.
    
        '''
        
        if trajectories_indices is None:
            trajectories_indices = self.trajectories_indices
        
        # to get the total time needed to write the data for all trajectories
        # Time needed to write the data
        # write_data_time = 0
        
        time_start = time.time()
        # Load user paths and add users to the scene    
        self.load_user_trajectories()
        print("Time taken to load the user trajectories and add users to the scene:", time.time() - time_start)
            
    
        if get_h_freq:# Subcarrier frequencies for the OFDM channel response
            scs = self.params["subcarrier_spacing"]
            fft = self.params["fft_size"]
            self.rb_frequencies = subcarrier_frequencies(fft, scs)[::24]  # 12SC=1RB
            
        
        # Process each trajectory in the list of trajectories_indices
        for trajectory_idx in tqdm(trajectories_indices, desc="Processing Trajectories"):
            
            remove_receivers(self.scene)
            # To track the time needed to write the data for the current trajectory
            write_data_time = 0
            trajectory_start_time = time.time()
            
            # Number of points in the trajectory            
            num_points = (len(self.user_trajectories_dict[trajectory_idx])) # Num points where user will be mooved
            
            # Verify the number of positions in parallel to be used is not superior to the number of positions in the path
            n_positions_in_parallel_ =min(n_positions_in_parallel,num_points)
            
            # Adding the users to the scene // from the first point in the path // n_positions_in_parallel_ users will be added
            self.add_users_to_scene(trajectory_idx,0,n_positions_in_parallel_)
            
            
        
            max_size_per_data_set = self.get_max_size_per_data_set(num_points)
        
            if self.verbose:
                print("Max size per data (over estimated) :",max_size_per_data_set)
                print("Number of points in path:", num_points, "Path index:", trajectory_idx)
            
            try:
                # Prealocating the data with the excpected size to store the data and then write the data at the excpected index
                self._preallocate_datasets(traj_idx=trajectory_idx,
                                           max_sizes=max_size_per_data_set
                    )
                
            except TrajectoryDatasetExistsException as e:
                print("Skipping the trajectory ",trajectory_idx)
                continue # skip the current trajectory and continue with the next one
            
            except Exception as e:
                print("Error while prealocating the data for trajectory ",trajectory_idx,"Exception",e)
                continue # skip the current trajectory and continue with the next one
            if self.verbose:
                    print("Max size per data :",max_size_per_data_set)


           # Itteration over the the diffeerent positions of the user in the path by step of batch_size*n_user_in_parallel
            for i in tqdm(range(0, (num_points), batch_size*n_positions_in_parallel_), desc=f"Processing Points in Path {trajectory_idx}"): 
                                
                # Those will have first dimension equalt to numpoint-1 * num_time_steps
                # Initialize batch variables -> positions and reshaped h_f
                if get_h_freq:
                    batch_csi = []          # 旧: batch_h_f
                batch_time_steps = []     
               
                
                # Process points inside the same batch : Itteration over the batch positions by the number of users to process in paralle 
                for j in range(i, min(i + batch_size*n_positions_in_parallel_, num_points),n_positions_in_parallel_): 

                    batch_time_steps.append(
                        self.time_steps_dict[trajectory_idx][j : j + n_positions_in_parallel_]
                                            )
                    
                    
                    # Update the users in the scene 
                    self.update_users_in_scene(trajectory_idx,j,n_positions_in_parallel_)
                    
                    # Compute paths 
                    paths = self.compute_scene_paths()
                    
                    # Normalize delays and apply Doppler effect
                    paths.normalize_delays = self.params["path.normalize_delays"]
                    
                    # get the a and tau values for the current point in the path 
                    # Size of a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
                    # Size of tau : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or :
                    # Size of tau : [batch_size, num_rx, num_tx,  max_num_paths]  if synthetic array is True
        
                    a, tau = paths.cir(self.params["cir_los"], self.params["cir_reflection"], self.params["cir_diffraction"],self.params["cir_scattering"], self.params["cir_ris"], self.params["cir_cluster_ris_paths"],num_paths=self.params["cir_num_paths"])
                    
                    # To limit the memory usage we can remove the paths after the computation of the CIR
                    paths = None
                    
                    if get_h_freq:
                        h_f = cir_to_ofdm_channel(self.rb_frequencies, a, tau,
                          normalize=self.params["cir_to_ofdm_normalize"])
                        h_f = tf.squeeze(h_f, axis=[5])                    # shape [bs,rx,ant_rx,tx,ant_tx,RB]
                        bs, rx, ant_rx, tx, ant_tx, rb = h_f.shape

                        h_f = tf.reshape(                                             # ← 追加：rx を time に吸収
                                h_f, [bs*rx, ant_rx, tx*ant_tx, rb])                 # [time,RxAnt,TxTot,RB]
                        h_f = tf.transpose(h_f, perm=[0, 3, 2, 1])                   # [time,RB,TxTot,RxAnt]


                        h_f = tf.stack([tf.math.real(h_f), tf.math.imag(h_f)], axis=-1)  # 実虚分離
                        batch_csi.append(h_f.numpy())
                    
                # Combine batch results // Concatenate the results of the batch over the first dimension
                # ----------<< New write block : ONE time per i-loop >>----------
                batch_csi        = np.concatenate(batch_csi, axis=0)        # [step,RB,TxTot,RxTot,2]
                batch_time_steps = np.concatenate(batch_time_steps, axis=0) # [step]

                batch_data = {
                    "csi":        batch_csi.astype(np.float32),
                    "timestamps": batch_time_steps.astype(np.float32)
                }

                self._write_to_dataset(traj_index=trajectory_idx,
                                    data_dict=batch_data,
                                    start_index=i)
                
                                
                # if self.verbose:
                #      print(f"saved batch @ step {i}, shape {batch_csi.shape}")
            
                # saving_batch_data_time = time.time() - saving_batch_data_time        
                # write_data_time += saving_batch_data_time
            
            # Includ the time needed to generate and save the data for the current trajectory
            trajectory_processing_time = time.time() - trajectory_start_time
            
            self.execution_time[trajectory_idx] = {"total_time": trajectory_processing_time, "write_data_time": write_data_time}
            
            # Print the data in a single line
            print(f"Trajectory {trajectory_idx}: Total Processing Time = {trajectory_processing_time:.4f} sec, Write Data Time = {write_data_time:.4f} sec")

            if self.verbose:
                print("############# TOTAL TIME taken to write the data for trajectory %d is %.2f seconds" % (trajectory_idx, write_data_time))
            
            if trajectory_idx in self.cached_datasets:
                # Deleting the cached datasets for the current path 
                del self.cached_datasets[trajectory_idx]

            remove_receivers(self.scene)            
            # End of the loop for the current trajectory
            
        # Closing finaly the file
        if self.hf is not None:
            self.hf.close()
            self.hf = None   
        
    # Prealocate numpy arrays with the excpected size to store the data // This will be done for each path /
    # --- PATCH ④ -----------------------------------------------
    def _preallocate_datasets(self, traj_idx, max_sizes,
                            output_dir="./out", prefix="csi"):

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{prefix}_traj_{traj_idx}.h5")
# ------------------------------------------------------------

        
        # Check if the file already exists and if the data for the current trajectory is already saved in the file
        with h5py.File(filepath, 'a',fs_persist=True,libver="latest") as hf:
                    
            if f"traj_{traj_idx}" in hf:    
                group = hf[f"traj_{traj_idx}"]
                all_trajectory_data_valid = True

                # Check if all datasets exist and are valid
                for dataset_name, max_size in max_sizes.items():
                    if dataset_name not in group:
                        all_trajectory_data_valid = False
                        break
                    else:
                        if group[dataset_name].shape[0] != max_size[0]:
                            all_trajectory_data_valid = False
                            break
                        else:
                            if np.isnan(group[dataset_name][:]).any():
                                all_trajectory_data_valid = False
                                break

                if all_trajectory_data_valid:  # All data is valid , no need to regenerate the data
                    print("All data is valid for trajectory ",traj_idx,". Skipping the trajectory.")
                    raise TrajectoryDatasetExistsException(f"Trajectory {traj_idx} data already exists in the file")
                
                else: # Nans found in the data
                    print("Deleting the group and recreating it, please re-run the data generation.")
                    del hf[f"traj_{traj_idx}"]
                    group = hf.create_group(f"traj_{traj_idx}") 
            else:
                # Create a group for the current trajectory  // here we can verify that the group is not already created (to avoid to generate the data again)
                try:
                    # Create a group for the current trajectory - if the group already exists, it will raise an value error
                    group = hf.create_group(f"traj_{traj_idx}")
                    all_trajectory_data_valid = False

                except ValueError as e: 
                    print(f"Error while creating the group for the trajectory {traj_idx}., Exception: {e}")
                    raise e
                
            # If the data is not valid or doesnt' exist we will create the data set and pre-allocate the data
            for dataset_name, max_size in max_sizes.items():
                # Parameters for the dataset creation
                chunk_size_bytes = self.chunk_sizes_per_data_set_bytes[dataset_name]
                num_chunks_in_dataset_cash = 10                                      # Number of chunks to cache in memory
                rdcc_nbytes = num_chunks_in_dataset_cash * chunk_size_bytes          # Number of bytes to use for the raw data chunk cache size
                rdcc_w0 = 0                                                          # Chunk preemption policy (which chunks to flush first)
                rdcc_nslots=101                                                      # Number of chunk slots in the raw data chunk cache 
                #  The number of chunk slots in the dataset’s chunk cache. Increasing this value reduces the number of cache collisions, but slightly increases the memory used                                                                         

                try:
                    group.create_dataset(
                        dataset_name, 
                        shape=max_size,
                        compression='lzf',                                          # Compression filter to use
                        shuffle=True,                                               #  Shuffle filter for better compression
                        chunks=self.chunk_size_per_data_set.get(dataset_name,True), # Use the predefined chunk size for the data set //
                        rdcc_nbytes=rdcc_nbytes,                                    # Number of bytes to use for the raw data chunk cache size
                        rdcc_nslots=rdcc_nslots,
                        fillvalue= self.data_types[dataset_name](np.nan),               # Fill value for the dataset
                        rdcc_w0=rdcc_w0,
                        dtype=self.data_types[dataset_name],
                    )
                    
                except Exception as e:  # Handle all errors
                    print(f"Issue creating dataset '{dataset_name}' for trajectory {traj_idx}.")
                    print(f" - Error details: {e}")   
                    raise e
                    
                if self.verbose:
                    print("INITIALIZATION OF THE DATA  FOR THE TRAJ ",traj_idx)
                    print(f"Dataset '{dataset_name}' created with max shape {max_size}")
                    print("Chunk size in bytes: ", self.chunk_sizes_per_data_set_bytes[dataset_name])
                    print("Chunk size",group[dataset_name].chunks)
                    print("Chunk size from the self.chunk_size_per_data_set",self.chunk_size_per_data_set.get(dataset_name))
                    print("########### Data pre-allocation done ")
            
            
        # To run only one read over the file and keep it as attibutes //
        self.hf = h5py.File(filepath,'a',libver='latest',fs_persist=True)
        
        # Keep the datasets in memory to avoid multiple reads // (we will use this cached datasets to write the data for the current trajectory)
        group = self.hf.require_group(f"traj_{traj_idx}")
        self.cached_datasets[traj_idx] = {key: group[key] for key in group.keys()}  # Cache datasets by trajectory

    # Write the data to the dataset at the given index
    def _write_to_dataset(self, traj_index, data_dict,start_index):
        """
        Writes data to a specified dataset at a given starting index.

        Args:
            group (h5py.Group): HDF5 group containing the dataset.
            dataset_name (str): Name of the dataset to write to.
            data (np.ndarray): Data to write into the dataset.
            start_index (int): The index at which to start writing the data.
        """        
        for key, value in data_dict.items():
            
            dataset = self.cached_datasets[traj_index][key]
            
            # Ensure the data fits into the datasets
            end_index = start_index + value.shape[0]
        
            try:
                dataset[start_index:end_index,...] = value
            
            except Exception as e:
                print("Error in the writing of the data :",key," for the path ",traj_index," at the index ",start_index," to ",end_index)
                print(f"Exception: {e}")
                raise e
class TrajectoryDatasetExistsException(Exception):
    pass


# Example usage
if __name__ == "__main__":

    # Path to the json file containing the simulation parameters
    params = load_dict_from_json("./Scenario/simulation_params.json")
    # Load the scene and set up radio materials
    scene = process_scene(params["scene_xml_fn"])
    
    # Load base station locations
    base_stations_locations_df = pd.read_csv(params["base_station_locations_fn"])
    base_stations_locations_df['z']=params['base_station_height']
    
    # Set up anntena array for receivers and transmitters
    setup_antenna_arrays(scene, params)
    
    # Add base stations to the scene
    add_base_stations_to_scene(scene, base_stations_locations_df, params["base_station_height"])

    # Set the frequency of the scene
    scene.frequency = params["scene_frequency"]
    
    # Set the synthetic array parameter for the scene    
    scene.synthetic_array = params["synthetic_array"]

    # read the meta_information of the different trajectories to select some of them
    trajectories_meta_info = pd.read_csv("./Scenario/0626_s200_straight/s200_straight_paths_random_start_goal_smoothed_meta_data.csv")
    
    trajectories_indices = list(trajectories_meta_info[(trajectories_meta_info["ration_invalid"] == 0)]["index"])

    # Shuffle the indices randomly with the fixed seed
    shuffled_indices = np.random.permutation(trajectories_indices)

    # Initialize the total path distance sum
    total_number_of_sampels = 0
    selected_indices = []
    
    # Iterate over the shuffled indices and accumulate distances
    for idx in shuffled_indices:
        # Get the 'original_path_distance' for the current index 
        original_path_distance = trajectories_meta_info.loc[trajectories_meta_info["index"] == idx, "original_path_distance"].values[0]
        # Get the number of points in the path
        total_number_of_sampels += int((original_path_distance / params["user_speed"]) * params["sampling_frequency"])
  
        # Append the index to the selected_indices list
        selected_indices.append(idx)
                
        # Check if the total number of samples is greater than or equal to 24 hours of simulation
        if total_number_of_sampels >= 3600 * 24 * params["sampling_frequency"]:
            break
    # To run the simulation for all the trajectories of 24 hours
    trajectories_indices = selected_indices
    data_set_file_name = f"csi_data_24h_trajectories.h5"
    
    print("Selected trajectories indices :",trajectories_indices)
    
    start = time.time()
    data_generator = DataGenerator(scene, params, trajectories_indices=trajectories_indices,verbose=False,get_h_freq=False)    
    
    print("Time taken to initialize the data generator:", time.time() - start)
    start = time.time()
    
    # Main function to generate the data for the selected trajectories
    data_generator.generate_cir_per_batch(trajectories_indices=trajectories_indices, 
                                          batch_size=10,
                                          n_positions_in_parallel=100,
                                          get_h_freq=True
                                          )
    
    end_time = time.time()
    # CSV file path and headers
    csv_file_path_for_execution_times = "execution_times.csv"
    headers = ["trajectory_idx", "total_time", "write_data_time"]

    # Gather all execution time data
    execution_times_data = [
        {
            "trajectory_idx": trajectory_idx,
            "total_time": times["total_time"],
            "write_data_time": times["write_data_time"]
        }
        for trajectory_idx, times in data_generator.execution_time.items()
    ]
    
    print("Time taken to generate the data:", end_time - start)
    
    # To save execution times to a csv file
    csv_file_path = "execution_times.csv"
    headers = ["trajectory_idx", "total_time", "write_data_time"]
        
    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        # Write headers only if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Directly write the execution times data
        writer.writerows([
            {"trajectory_idx": trajectory_idx, **times}
            for trajectory_idx, times in data_generator.execution_time.items()
        ])

      
    print(f"Execution times saved to {csv_file_path}")
