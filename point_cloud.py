import os
import cv2
import open3d as o3d
import numpy as np
from tqdm import tqdm

# Set paths
rgb_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_images_w"
depth_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/depth_maps_w"
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/point_clouds_w"
os.makedirs(output_folder, exist_ok=True)

# Process all images
for file in tqdm(os.listdir(rgb_folder)):
    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    name = os.path.splitext(file)[0]
    rgb_path = os.path.join(rgb_folder, file)
    depth_path = os.path.join(depth_folder, name + "_depth.png")
    output_path = os.path.join(output_folder, name + ".ply")

    if not os.path.exists(depth_path):
        print(f"[!] Missing depth map for {file}")
        continue

    # Load images
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

    # Convert to Open3D images
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=255.0,  # adjust if needed
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    width, height = rgb.shape[1], rgb.shape[0]
    focal_length = width
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, width / 2, height / 2)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Save to file
    o3d.io.write_point_cloud(output_path, pcd)

print("Batch point cloud generation complete.")

'''import open3d as o3d

# Load saved point cloud
pcd = o3d.io.read_point_cloud("/Users/edelta076/Desktop/Project_VID_Assistant2/point_clouds/716.ply")
print(pcd)  # Should print number of points > 0
# Visualize
o3d.visualization.draw_geometries([pcd])'''

'''import open3d as o3d
mesh = o3d.io.read_triangle_mesh("/Users/edelta076/Desktop/Project_VID_Assistant2/ed_mesh/716_mesh.glb")
o3d.visualization.draw_geometries([mesh])'''

