import open3d as o3d
import os
from tqdm import tqdm
import numpy as np

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

        # Skip empty point clouds
        if len(pcd.points) == 0:
            print(f"⚠️ Skipping empty point cloud: {file}")
            continue

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Check if normals are valid
        if not pcd.has_normals():
            print(f" Normals not computed for: {file}")
            continue

        # Ball Pivoting Reconstruction (works better than Poisson on macOS)
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        if len(mesh.triangles) == 0:
            print(f" No mesh triangles generated for: {file}")
            continue

        # Save mesh
        o3d.io.write_triangle_mesh(os.path.join(mesh_folder, f"{name}_mesh.glb"), mesh)

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Finished generating meshes.")
