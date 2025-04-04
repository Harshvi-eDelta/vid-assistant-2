import open3d as o3d
import os
from tqdm import tqdm

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
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Ball Pivoting (stable mesh reconstruction)
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        # Save as OBJ (GLB disabled to avoid crash)
        o3d.io.write_triangle_mesh(os.path.join(mesh_folder, f"{name}_mesh.obj"), mesh)

    except Exception as e:
        print(f"⚠️ Failed to process {file}: {e}")
