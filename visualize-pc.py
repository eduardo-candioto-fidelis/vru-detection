import numpy as np
import open3d as o3d

# Load point cloud data from npy file
point_cloud_data = np.load('dataset/points/000081.npy')
points = point_cloud_data[:, :3]  # Extract x, y, z coordinates

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Set a uniform color for all points in the point cloud (e.g., white)
colors = np.ones_like(points)  # Create an array of ones (white color)
colors[:, 2] = 1
colors[:, 0] = 0.1
colors[:, 1] = 0.1
pcd.colors = o3d.utility.Vector3dVector(colors)

# Load bounding boxes from txt file
bounding_boxes = []
with open('dataset/labels/000081.txt', 'r') as f:
    for line in f:
        data = line.strip().split()
        x, y, z, dx, dy, dz, heading_angle, category_name = data[:8]
        x, y, z = float(x), float(y), float(z)
        dx, dy, dz = float(dx), float(dy), float(dz)
        heading_angle = float(heading_angle)
        category_name = category_name
        
        # Create a bounding box
        center = [x, y, z]
        extent = [dx, dy, dz]
        rotation_matrix = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, heading_angle])
        bbox = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
        
        # Set the color of the bounding box to blue
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color([1, 0, 0])  # RGB for blue
        
        bounding_boxes.append((bbox, lineset))

# Initialize the colors of the points to white
#colors = np.ones_like(points)

# Check if each point is inside any of the bounding boxes
for bbox, _ in bounding_boxes:
    mask = bbox.get_point_indices_within_bounding_box(pcd.points)
    colors[mask] = [1, 1, 1]  # Set the color to red for points inside any bounding box

# Assign the colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)

# Create a coordinate frame
axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Create a visualization object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud, bounding boxes, and coordinate frame to the visualization
vis.add_geometry(pcd)
for _, lineset in bounding_boxes:
    vis.add_geometry(lineset)
vis.add_geometry(axis_frame)

# Set the background color to black
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])

# Reduce the size of the points
opt.point_size = 2.0  # Set the point size (try smaller values if necessary)

# Run the visualization
vis.run()
vis.destroy_window()
