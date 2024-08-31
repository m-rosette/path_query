#!/usr/bin/env python3

import numpy as np

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from apple_approach_interfaces.srv import CoordinateToTrajectory, SendTrajectory


class CoordinateToTrajectoryService(Node):
    def __init__(self):
        super().__init__('voxel_search_service')
        
        # Create the service
        self.srv = self.create_service(CoordinateToTrajectory, 'coordinate_to_trajectory', self.coord_to_traj_callback)

        # Create the service client
        self.client = self.create_client(SendTrajectory, 'execute_arm_trajectory')

        # Create a publisher for MarkerArray
        self.marker_publisher = self.create_publisher(MarkerArray, 'voxel_markers', 10)

        # Set the timer to publish markers periodically
        self.timer = self.create_timer(1.0, self.publish_markers)

        # Define distance tolerance
        self.distance_tol = 1.0

        # Load voxel data
        y_trans = 0.45
        self.voxel_data = np.loadtxt('/home/marcus/ros2_ws/src/path_query/apple_approach/resources/voxel_data_parallelepiped_filtered.csv')
        self.paths = np.load('/home/marcus/ros2_ws/src/path_query/apple_approach/resources/voxel_paths_parallelepiped.npy')
        self.load_voxel_data(y_trans)

        self.get_logger().info('Coordinate to trajectory service up and running')

    def load_voxel_data(self, y_translation):
        self.voxel_centers = self.voxel_data[:, :3]
        self.voxel_indices = self.voxel_data[:, 3:]

        # Translate voxels in front of robot
        voxel_centers_shifted = np.copy(self.voxel_centers)
        voxel_centers_shifted[:, 1] += y_translation
        self.voxel_centers = voxel_centers_shifted

    def publish_markers(self):
        marker_array = MarkerArray()

        for i, center in enumerate(self.voxel_centers):
            marker = Marker()
            marker.header.frame_id = 'base_link'  # Change this to your fixed frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'voxel'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Create and set the Point object
            point = Point()
            point.x = center[0]
            point.y = center[1]
            point.z = center[2]
            marker.pose.position = point
            
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # Radius of the sphere
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully opaque

            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

    def coord_to_traj_callback(self, request, response):
        # Extract the requested coordinate
        point_msg = request.coordinate
        x = point_msg.x
        y = point_msg.y
        z = point_msg.z
        point = np.array([x, y, z])

        # Find the closest path associated with the target point
        path, distance_to_voxel = self.path_to_closest_voxel(point)
        self.get_logger().info(f'Distance to nearest voxel: {distance_to_voxel}')

        if distance_to_voxel > self.distance_tol:
            self.get_logger().error("Distance to nearest voxel exceeded the distance threshold. Cancelling trajectory execution...")
            response.success = False
        
        else:
            # Package path as a float multiarray
            float32_array = Float32MultiArray()

            # Set the layout (optional, but helps with multi-dimensional arrays)
            float32_array.layout.dim.append(MultiArrayDimension())
            float32_array.layout.dim[0].label = "rows"
            float32_array.layout.dim[0].size = path.shape[0]
            float32_array.layout.dim[0].stride = path.size

            float32_array.layout.dim.append(MultiArrayDimension())
            float32_array.layout.dim[1].label = "columns"
            float32_array.layout.dim[1].size = path.shape[1]
            float32_array.layout.dim[1].stride = path.shape[1]

            # Flatten the NumPy array and assign it to the data field
            float32_array.data = path.flatten().tolist()

            # Assign to the response
            response.success = True

        if response.success:
            self.waypoint_msg = float32_array
            self.trigger_arm_mover()

        return response

    def trigger_arm_mover(self):
        if not self.client.service_is_ready():
            self.get_logger().info('Waiting for execute_arm_trajectory service to be available...')
            self.client.wait_for_service()

        request = SendTrajectory.Request()
        request.waypoints = self.waypoint_msg  # Pass the entire Float32MultiArray message

        # Use async call
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_trajectory_response)

    def handle_trajectory_response(self, future):
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info('Waypoint path service call succeeded')
            else:
                self.get_logger().error('Waypoint path service call failed')
        except Exception as e:
            self.get_logger().error(f'Exception occurred: {e}')

        # Reset state after the service call
        self.waypoint_msg = None

    def path_to_closest_voxel(self, target_point):
        """ Find the path to a voxel that the target point is closest to 

        Args:
            target_point (float list): target 3D coordinate

        Returns:
            path: the path to the voxel the target point is closest to
            distance_error: error between target point and closest voxel center
        """
        # Calculate distances
        distances = np.linalg.norm(self.voxel_centers - target_point, axis=1)
        
        # Find the index of the closest voxel
        closest_voxel_index = np.argmin(distances)

        distance_error = distances[closest_voxel_index]

        # Get the associated path to closest voxel
        return self.paths[:, :, closest_voxel_index], distance_error
        

def main():
    rclpy.init()

    coord_to_traj = CoordinateToTrajectoryService()

    # Use a SingleThreadedExecutor to handle the callbacks
    executor = SingleThreadedExecutor()
    executor.add_node(coord_to_traj)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        coord_to_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()