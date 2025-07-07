"""
 my_probabilistic_road_map.py
"""

import mitsuba as mi
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import heapq
# To use the Mitsuba renderer with the scalar_rgb variant (on CPU)
mi.set_variant('scalar_rgb')
# Seed NumPy's random module
# np.random.seed(42)
def smooth_paths(paths, sampling_frequency, scene, user_speed=1.0, z_offset=0.1):
    """
    Resample each raw path with a fixed temporal resolution (1/sampling_frequency [s]).
    Returned paths have *variable* length depending on their travel time.
    """
    from scipy.interpolate import CubicSpline
    smoothed_paths, meta = [], {}

    for idx, raw in enumerate(paths):
        raw = np.asarray(raw)

        # ── 移動距離と所要時間 ───────────────────────────
        segment_len = np.linalg.norm(np.diff(raw, axis=0), axis=1)


        total_dist  = segment_len.sum()
        total_time  = total_dist / user_speed          # [s]

        # ── 時刻列（等間隔）─────────────────────────────
        num_samples = max(int(np.round(total_time * sampling_frequency)), 2)
        t_samples   = np.linspace(0.0, total_time, num_samples)

        # ── 元データの「時刻」軸を作る（距離 / 速度） ──
        t_raw = np.insert(np.cumsum(segment_len / user_speed), 0, 0.0)

        # ── 各軸をスプライン補間し再サンプリング ────────
        splines = [CubicSpline(t_raw, raw[:,k], bc_type="natural") for k in range(3)]
        path    = np.vstack([s(t_samples) for s in splines]).T   # shape=(num_samples,3)


        
        # === 追加：補間後の距離を計算 =====================
        smoothed_seg_len   = np.linalg.norm(np.diff(path, axis=0), axis=1)  # ← 変更
        smoothed_total_dist = smoothed_seg_len.sum()                        # ← 変更
        # ==================================================

        # ── 統計情報 ────────────────────────────────────
        inst_speed = smoothed_seg_len * sampling_frequency  # 最後は 0 でパディング

        invalid = sum(is_position_inside_building(scene, p, z_offset)
                  for p in path)
        ration   = invalid / len(path)


        meta[idx]  = {"average_speed": inst_speed.mean(),
                      "std_speed":     inst_speed.std(),
                      "total_distance":     smoothed_total_dist,
                      "invalid_points": invalid,
                      "ration_invalid":  ration,
                      "original_path_distance": total_dist            
                      }

        smoothed_paths.append(path)

    return smoothed_paths, meta

def is_position_inside_building(scene, position, z_offset=0.1):
    """
    Checks if a given position is within a building by casting ray upward from the specified position.

    Parameters:
    - scene: Mitsuba scene object.
    - position: tuple of (x, y, z) coordinates in world space.
    - z_offset: float , a decay to take off form the user z position.

    Returns:
    - bool: True if the position is within a building, False otherwise.
    """
    x, y, z = position
    
    # Handl cases where the user may be in low alltitude or in negative one !
    new_z = max(float(0),z-z_offset)
    # Cast an upward ray starting slightly below the position
    ray_origin_up = [x, y, new_z]
    ray_direction_up = [0, 0, 1]
    ray_up = mi.Ray3f(ray_origin_up, ray_direction_up)
    si_up = scene.ray_intersect(ray_up)
    
    # Check the validity of intersections and ignore ground plane ("mesh-Plane")
    building_above = si_up.is_valid() 

    # Determine if the position is within a building based on the ray results
    if building_above:
        # The point is within a building (both rays intersect buildings)
        return True
    else:
        # Not within a building if there's no structure both above and below
        return False
    
def sample_free_points_continuous(scene, num_samples, z_position=1.5, z_offset=0.1,min_bound=None,max_bound=None):
    """
    Sample random valid positions outside buildings within the bounding box of the scene or within the specified bounds.

    Parameters:
    - scene: Mitsuba scene object.
    - num_samples: int, the number of valid points to sample.
    - z_position: float, fixed z-coordinate to use for sampling (ground level).
    - z_offset: float, offset for checking if a position is inside a building.
    - min_bound : tuple (min_x,min_y,min_z) to condition the values from where we will sample the positions
    - max_bound : tuple (max_x, max_y, max_z)  to condition the region from where we will sample the positions

    Returns:
    - List of (x, y, z) tuples representing sampled positions outside buildings.
    """
    if min_bound is None and max_bound is None:
        
        # Get the bounding box of theF scene
        min_bound = np.array(scene.bbox().min)
        max_bound = np.array(scene.bbox().max)
    
    if z_position is not None:
        # Set the z-coordinate for sampling
        min_bound[2] = z_position
        max_bound[2] = z_position

    valid_points = []

    while len(valid_points) < num_samples:
        # Sample a random point in the bounding box
        sample_point = np.random.uniform(min_bound, max_bound)
        
        # Check if the point is inside a building
        if not is_position_inside_building(scene, sample_point, z_offset):
            valid_points.append(tuple(sample_point))
    
    return valid_points

def is_direct_path(scene, point_a, point_b):
    """
    Check if there is a direct, collision-free path between two points in 3D space.

    Parameters:
    - scene: Mitsuba scene object.
    - point_a, point_b: tuple, coordinates (x, y, z) of the start and end points.

    Returns:
    - bool: True if there is a direct path (no collision), False if there is an obstacle.
    """
    # Define the ray from point A to point B
    ray_origin = mi.Vector3f(point_a)
    direction_vector = mi.Vector3f(point_b) - ray_origin

    norm = (direction_vector.x**2 + direction_vector.y**2 + direction_vector.z**2)**0.5
    ray_direction = direction_vector / norm  # Normalized direction vector

    # Create a ray with the computed origin, direction, and length
    ray_length = norm  # The ray length is the norm of the direction vector
    ray = mi.Ray3f(ray_origin, ray_direction, ray_length)

    # Intersect the ray with the scene
    si = scene.ray_intersect(ray)
    
    
    # Check if there is an intersection within the ray's distance to point B
    if si.is_valid() and  si.t < ray_length:
        return False  # Obstacle found, no direct path
    
    return True  # No obstacles, direct path is clear

def sample_start_goal(scene, num_pairs, z_position=1.5, z_offset=0.1):
    """
    Select distinct start-end pairs sampled directly from continuous space, based on Euclidean distance.

    Parameters:
    - scene: Mitsuba scene object.
    - num_pairs: int, number of start-end pairs to generate.
    - z_position: float, fixed z-coordinate for sampling (ground level).
    - z_offset: float, offset for building occupancy check.

    Returns:
    - List of tuples with start and end points.
    """
    pairs = []
    start_list = sample_free_points_continuous(scene, num_pairs, z_position, z_offset)
    goals_list = sample_free_points_continuous(scene, num_pairs, z_position, z_offset)
    
    for i in range(num_pairs):    
        pairs.append((start_list[i], goals_list[i]))
        
    return pairs


def build_roadmap(free_points, scene, k=5):
    """
    Build a roadmap by connecting each point to its k nearest neighbors if collision-free.
    
    Parameters:
    - free_points: list of tuples, sampled points in real-world coordinates.
    - scene: Mitsuba scene object, used for collision checking between points.
    - k: int, number of nearest neighbors to connect.

    Returns:
    - roadmap: dict, adjacency list representation of the graph.
    - kdtree: KDTree, spatial index for nearest neighbor search.
    """
    roadmap = {point: [] for point in free_points}
    kdtree = KDTree(free_points)

    for point in free_points:
        # Find k nearest neighbors of the point (excluding itself)
        _, neighbors = kdtree.query(point, k + 1,p=2)  # +1 to exclude itself
        
        for i in range(1, len(neighbors)):  # Skip the first one (itself)
            neighbor = free_points[neighbors[i]]
            # Check if there is a direct path between the points (no collision)
            if is_direct_path(scene, point, neighbor):
                # If the path is collision-free, add the neighbor to the roadmap
                roadmap[point].append(neighbor)
                roadmap[neighbor].append(point) 
    return roadmap, kdtree


def heuristic(p1, p2):
    """Heuristic function for A* (Euclidean distance in 3D space)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def astar_search(roadmap, start, goal):
    """
    Perform A* search on the roadmap from start to goal.
    
    Parameters:
    - roadmap: dict, adjacency list representation of the graph.
    - start, goal: tuple, start and goal points in real-world coordinates.

    Returns:
    - path: list of tuples, path from start to goal.
    """ 
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        for neighbor in roadmap[current]:
            new_cost = cost_so_far[current] + heuristic(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    return None  # No path found

def connect_point_to_roadmap(point, roadmap, kdtree, scene, k=8):
    """
    Connect a point to its k nearest neighbors in the roadmap, based on the A* search and direct path validation.
    
    Parameters:
    - point: tuple, the point to connect.
    - roadmap: dict, the adjacency list representing the graph.
    - kdtree: KDTree, spatial index for nearest neighbor search.
    - scene: Mitsuba scene object, used for collision checking.
    - k: int, number of nearest neighbors to connect to.
    
    Returns:
    - roadmap: dict, updated roadmap with the new connections.
    """
    if roadmap.get(point) is None:
        
        roadmap[point] = []
        _, indices = kdtree.query(point, k)

        for idx in indices:
            neighbor = kdtree.data[idx]
            neighbor = tuple(neighbor)

            # Check if a direct path exists between the current point and the neighbor
            if is_direct_path(scene, point, neighbor):
                # If there's no obstacle, add the neighbor to the roadmap
                roadmap[point].append(neighbor)
                roadmap[neighbor].append(point)  
                
        # Return the updated roadmap and The KDTree
        return roadmap,KDTree(list(roadmap.keys()))
    else:
        return roadmap,kdtree


    
if __name__=="__main__":
    
    # Code to generate the trajectories using PRM and A* search    
    # Parameters
    K = 15                  # K nearest neighbors to connect in the roadmap
    Z = 1.5                 # Fixed Z position
    Z_OFFSET =0            # Offset for checking if a position is inside a building (e.g., 0.1) 
    N = 10000                # Number of valid positions to sample
    NUMBER_OF_VARIANTS = 1000 # Number of pairs to generate (start and goal)
    start = (-30.0,10,Z)
    waypoints  = [(-30, 10, Z), 
                  #( -27.222, 1.111, Z),
                  (-21.667,	-16.667, Z),
                  #(-21.667 , -16.667, Z),
                  ( -13.333,	-43.333 , Z),
                  #( -10.556, -52.222, Z),
                  #( -5.0,-70.0, Z)
                  ] 

    goal = (-5,-70,Z)
    routing_points_template = [start] + waypoints + [goal]
    
    #start_goal_paris = [(start, goal)]*NUMBER_OF_VARIANTS

    START_REGION_HEIGHT= 50 # Height of the starting region
    MAX_TRIES = 5           # Maximum number of tries to find a path      


    scene = mi.load_file('./Scenario/citi_last.xml')  # Load your scene XML file
    bbox = scene.bbox()
    min_x, min_y = bbox.min[0],bbox.min[1]
    max_x, max_y = bbox.max[0],bbox.max[1]


    # To generate a topographic map // Sample uniformly the points in the scene at a given fixed height
    Z = 1.5
    NUM_SAMPLES_TOPOGRAPHIC_MAP = 100000
    
    topographic_map_data = sample_free_points_continuous(scene, num_samples=NUM_SAMPLES_TOPOGRAPHIC_MAP,z_position=Z)

    # Visualize the topographic map if needed and save it
    # plt.figure(figsize=(10, 10))
    # plt.scatter(*zip(*topographic_map_data))    
    # plt.title('Topographic Map of the Scene')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    
    # Probabilistic Road Map (PRM)     
    # To get points at different heights, set different z components for the min and max bounds
    # Defining a starting and ending regions width
    starting_ending_region_width = 10

    #start_goal_paris = sample_start_goal(scene, NUMBER_OF_PAIRS)

    paths = []
    distances = []

    # Sample positions only once 
    # valid_positions = sample_free_points_continuous(scene, N,z_position=Z,z_offset=Z_OFFSET)
    
    # for i, (start, goal) in enumerate(start_goal_paris):

    #     np.random.seed(i)
    #     N_variant = N + np.random.randint(-50, 51)
    #     K_variant = K + np.random.randint(-2, 3) 
        
    #     path = None
    #     nb_tries = 0
        
    #     while path is None and nb_tries < MAX_TRIES:
            
    #         # Sampling N points for each trajectory
    #         # Sample N points :
    #         valid_positions = sample_free_points_continuous(scene, N_variant,z_position=Z,z_offset=Z_OFFSET)
            
    #         # Build roadmap=
    #         #roadmap,kdtree = build_roadmap(valid_positions,scene,k=K)

    #         roadmap, kdtree  = build_roadmap(valid_positions, scene, k=K_variant)
    #         # Add the start and goal points in the roadmap in case that they were not sampled // connect them to the roadmap
    #         roadmap,kdtree = connect_point_to_roadmap(start,roadmap,kdtree,scene,K_variant )
    #         roadmap,kdtree = connect_point_to_roadmap(goal,roadmap,kdtree,scene,K_variant )

    #         path = astar_search(roadmap,start,goal)
    #         nb_tries += 1


    for i in range(NUMBER_OF_VARIANTS):

        np.random.seed(i)
        N_variant = N + np.random.randint(-50, 51)
        K_variant = K + np.random.randint(-2, 3)

        full_path  = None          # ← 経由点すべてを通る最終経路
        nb_tries   = 0

        while full_path is None and nb_tries < MAX_TRIES:
            valid_positions = sample_free_points_continuous(scene, N_variant,
                                                            z_position=Z, z_offset=Z_OFFSET)

            roadmap, kdtree = build_roadmap(valid_positions, scene, k=K_variant)

            # --- 経由点を含む全ポイントをロードマップへ接続 -------------
            routing_points = routing_points_template.copy()
            for p in routing_points:
                roadmap, kdtree = connect_point_to_roadmap(p, roadmap, kdtree, scene, K_variant)
            # -----------------------------------------------------------

            # === ④ 区間ごとに A* を実行して連結 ========================
            full_path, path_found = [], True
            for s, g in zip(routing_points[:-1], routing_points[1:]):   # (p0→p1), (p1→p2)… 
                segment = astar_search(roadmap, s, g)
                if segment is None:
                    path_found = False
                    break
                full_path.extend(segment if not full_path else segment[1:])
            # ===========================================================

            if not path_found:
                full_path = None
            nb_tries += 1

        
        # if path is None:
        #     print(f"Failed to find a path between {start} and {goal}")
        #     continue
        # else:
        #     path_distance = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path)-1))
        #     print("Path found : Nb points",len(path),"Path total distance : ",path_distance)
            
        # paths.append(path)
        # distances.append(path_distance)

        if full_path is None:
            print("Failed to find a full path through all waypoints")
            continue
        else:
            path_distance = sum(
                np.linalg.norm(np.array(full_path[i]) - np.array(full_path[i+1]))
                for i in range(len(full_path)-1))
            print("Path found : Nb points", len(full_path),
                "Path total distance :", path_distance)

        paths.append(full_path)
        distances.append(path_distance)


    # Save original paths :
    flattened_paths = []

    # Iterate through each path and add data to the flattened list
    for path_index, path in enumerate(paths):
        # Extract x, y, z coordinates for each point in the path and append to the list
        flattened_paths.extend([[path_index, x, y, z] for (x, y, z) in path])

    # Convert to a DataFrame
    df = pd.DataFrame(flattened_paths, columns=["index", "x", "y", "z"],index=None)

    # Save DataFrame to CSV
    output_file_path = "./Scenario/0627_s4_straight/s4_straight_paths_random_start_goal_no_smooth.csv"
    df.to_csv(output_file_path, index=False)
    
    # Smooth the paths main parameters
    sampling_frequency  = 4
    user_speed = 1.0

    # Smooth the paths and get some meta data
    smoothed_paths,smoothed_paths_meta_data =  smooth_paths(paths, sampling_frequency,scene, user_speed=1.0 )
    
    # Save smoothed paths meta data //
    paths_meta_data_df = pd.DataFrame([
    {"index": idx, **data} for idx, data in smoothed_paths_meta_data.items()])
    
    paths_meta_data_df.to_csv("./Scenario/0627_s4_straight/s4_straight_paths_random_start_goal_smoothed_meta_data.csv", index=False)
    
    # Save smoothed paths into a CSV file
    flattened_paths = []

    # Iterate through each path and add data to the flattened list
    for path_index, path in enumerate(smoothed_paths):
        # Extract x, y, z coordinates for each point in the path and append to the list
        flattened_paths.extend([[path_index, x, y, z] for (x, y, z) in path])

    # Convert to a DataFrame
    df = pd.DataFrame(flattened_paths, columns=["index", "x", "y", "z"],index=None)

    df.to_csv("./Scenario/0627_s4_straight/s4_straight_paths_random_start_goal_smoothed.csv", index=False)

        
