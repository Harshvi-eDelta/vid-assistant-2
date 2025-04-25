import bpy
import sys
import os

# --- CONFIG --- #
input_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/3d_mesh"     # Change this
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/output_uv_meshes"  # Will save UV-unwrapped versions
os.makedirs(output_folder, exist_ok=True)

# --- GET FILES --- #
mesh_files = [f for f in os.listdir(input_folder) if f.endswith((".obj", ".glb", ".gltf"))]

for file in mesh_files:
    input_path = os.path.join(input_folder, file)
    name, ext = os.path.splitext(file)
    output_path = os.path.join(output_folder, f"{name}_uv{ext}")

     # --- RESET SCENE --- #
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- IMPORT MESH --- #
    ext = ext.lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=input_path)
    elif ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        print(f"Unsupported file format: {ext}")
        continue

    # --- GET MESH OBJECT --- #
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # --- UV UNWRAP --- #
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- EXPORT WITH UV --- #
    if output_path.endswith(".obj"):
        bpy.ops.export_scene.obj(filepath=output_path, use_uvs=True)
    elif output_path.endswith(".glb"):
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    else:
        print("Unsupported export format")
        continue

    print(f" UV unwrapped mesh saved: {output_path}")

print(" All meshes processed.")


# to tun this on terminal 

'''
blender --background --python uv_unwrap_batch.py

'''