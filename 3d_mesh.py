'''import open3d as o3d
import os
from tqdm import tqdm
import numpy as np

# Input and output folders
pcd_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/point_clouds"
mesh_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/3d_mesh"
os.makedirs(mesh_folder, exist_ok=True)

for file in tqdm(os.listdir(pcd_folder)):
    if not file.endswith(".ply"):
        continue

    name = os.path.splitext(file)[0]
    pcd_path = os.path.join(pcd_folder, file)

    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)

        if len(pcd.points) == 0:
            print(f"Skipping empty point cloud: {file}")
            continue

        # Optional: Remove noise
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(50)

        # Ball Pivoting (BPA) reconstruction
        radii = [0.003, 0.005, 0.01, 0.02]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        if len(mesh.triangles) == 0:
            print(f" No mesh triangles generated for: {file}")
            continue

        # Clean mesh
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # Optional: Smooth mesh
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)

        # Visual inspection (optional)
        # o3d.visualization.draw_geometries([mesh])

        # Save mesh as GLB (works well with textures later)
        mesh_path = os.path.join(mesh_folder, f"{name}_mesh.glb")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Saved: {mesh_path}")

    except Exception as e:
        print(f" Error processing {file}: {e}")

print(" Finished generating all meshes.")'''

import open3d as o3d
import os
import time
import numpy as np
from tqdm import tqdm

pcd_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/point_clouds"
mesh_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/3d_mesh"
os.makedirs(mesh_folder, exist_ok=True)

MAX_FILES_PER_BATCH = 100

def process_batch():
    count = 0
    files = sorted(os.listdir(pcd_folder))
    for file in files:
        if not file.endswith(".ply"):
            continue

        name = os.path.splitext(file)[0]
        pcd_path = os.path.join(pcd_folder, file)
        mesh_path = os.path.join(mesh_folder, f"{name}_mesh.glb")

        if os.path.exists(mesh_path):
            continue  # already processed

        try:
            print(f"Processing: {file}")
            pcd = o3d.io.read_point_cloud(pcd_path)

            if len(pcd.points) == 0:
                print(f"Skipping empty point cloud: {file}")
                continue

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(50)

            radii = [0.003, 0.005, 0.01, 0.02]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

            if len(mesh.triangles) == 0:
                print(f"No mesh triangles generated for: {file}")
                continue

            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)

            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Saved: {mesh_path}")
            count += 1

            if count >= MAX_FILES_PER_BATCH:
                print(f"Processed {count} files in this batch.")
                return True  # batch complete

        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return False  # all done

# Loop until all are processed
while True:
    has_more = process_batch()
    if not has_more:
        print("All point clouds processed.")
        break
    print(" Sleeping before next batch...")
    time.sleep(10)  # Reduce CPU stress

