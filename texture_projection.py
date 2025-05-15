import open3d as o3d
import numpy as np
import cv2
import os

# === Input paths ===
rgb_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/original_jpg/1.jpg"
depth_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/output_depth_map_1.png"
output_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/output_textured_mesh_1.glb"

# === Load images ===
rgb = cv2.imread(rgb_path)
if rgb is None:
    raise ValueError("Could not load RGB image at " + rgb_path)

depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if depth is None:
    raise ValueError("Could not load depth image at " + depth_path)

# Resize depth to match RGB if necessary
if depth.shape != rgb.shape[:2]:
    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

# Convert to Open3D format
color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))

# Approximate camera intrinsics
height, width = rgb.shape[:2]
focal = width  # fx = fy = width
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width, height, focal, focal, width / 2, height / 2
)

# Create RGBD image and point cloud
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,
    depth_scale=255.0,
    depth_trunc=3.0,
    convert_rgb_to_intensity=False
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Mesh reconstruction
pcd.estimate_normals()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]
mesh.compute_vertex_normals()

# Assign per-vertex color using RGB image projection
colors = []
for v in np.asarray(mesh.vertices):
    x = int((v[0] * focal) / v[2] + width / 2)
    y = int((v[1] * focal) / v[2] + height / 2)
    if 0 <= x < width and 0 <= y < height:
        color = rgb[y, x] / 255.0
        colors.append(color[::-1])  # RGB
    else:
        colors.append([0.5, 0.5, 0.5])  # Gray fallback

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# Save textured mesh
o3d.io.write_triangle_mesh(output_path, mesh)
print(f" Textured mesh saved to: {output_path}")
