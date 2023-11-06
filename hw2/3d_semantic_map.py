import multiprocessing
from collections import Counter
import numpy as np
import open3d as o3d


def custom_voxel_down(pcd, voxel_size):
    def process_chunk(points_chunk, colors_chunk, voxel_size, result_queue):
        dc = {}
        for i in range(len(points_chunk)):
            x_idx = int(points_chunk[i][0] // voxel_size)
            y_idx = int(points_chunk[i][1] // voxel_size)
            z_idx = int(points_chunk[i][2] // voxel_size)
            voxel_idx = (x_idx, y_idx, z_idx)
            color_key = tuple(colors_chunk[i])
            if voxel_idx not in dc:
                dc[voxel_idx] = Counter()
            dc[voxel_idx][color_key] += 1
        result_queue.put_nowait(dc)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    num_processes = multiprocessing.cpu_count()
    chunk_size = len(points) // num_processes

    result_queue = multiprocessing.Queue()
    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(points)
        points_chunk = points[start_idx:end_idx]
        colors_chunk = colors[start_idx:end_idx]
        process = multiprocessing.Process(
            target=process_chunk,
            args=(points_chunk, colors_chunk, voxel_size, result_queue))
        processes.append(process)
        process.start()

    dc = {}
    cnt = 0
    while cnt < num_processes:
        if result_queue.empty():
            continue
        chunk_dc = result_queue.get()
        for voxel_idx, color_counter in chunk_dc.items():
            if voxel_idx not in dc:
                dc[voxel_idx] = Counter()
            dc[voxel_idx] += color_counter
        cnt += 1

    for process in processes:
        process.join()

    voxel_points = np.empty((len(dc), 3))
    voxel_colors = np.empty((len(dc), 3))
    for i, (voxel_idx, color_counter) in enumerate(dc.items()):
        voxel_colors[i] = color_counter.most_common(1)[0][0]
        voxel_points[i][0] = voxel_idx[0] * voxel_size + voxel_size / 2
        voxel_points[i][1] = voxel_idx[1] * voxel_size + voxel_size / 2
        voxel_points[i][2] = voxel_idx[2] * voxel_size + voxel_size / 2

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(voxel_points)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_colors)
    return pcd_down


""" Single process version, for comparison """


def custom_voxel_down_singleproc(pcd, voxel_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    voxel_indices = np.floor(points / voxel_size).astype(int)

    dc = {}
    for i in range(len(points)):
        voxel_idx = tuple(voxel_indices[i])
        color_key = tuple(colors[i])
        if voxel_idx not in dc:
            dc[voxel_idx] = Counter()
        dc[voxel_idx][color_key] += 1

    voxel_points = np.empty((len(dc), 3))
    voxel_colors = np.empty((len(dc), 3))
    for i, (voxel_idx, color_counter) in enumerate(dc.items()):
        voxel_colors[i] = color_counter.most_common(1)[0][0]
        voxel_points[i][0] = voxel_idx[0] * voxel_size + voxel_size / 2
        voxel_points[i][1] = voxel_idx[1] * voxel_size + voxel_size / 2
        voxel_points[i][2] = voxel_idx[2] * voxel_size + voxel_size / 2

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(voxel_points)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_colors)
    return pcd_down
