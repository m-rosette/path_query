#!/usr/bin/env python3

import numpy as np

import rclpy
import rclpy.logging
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from apple_approach_interfaces.srv import CoordinateToTrajectory, SendTrajectory


class CoordinateToTrajectoryService(Node):
    def __init__(self):
        super().__init__('voxel_search_service')
        
        # Create the service
        self.srv = self.create_service(CoordinateToTrajectory, 'coordinate_to_trajectory', self.coord_to_traj_callback)

        # Create the service client
        self.client = self.create_client(SendTrajectory, 'execute_arm_trajectory')

        self.voxel_data = np.loadtxt('/home/marcus/ros2_ws/src/apple_approach/resources/voxel_data_parallelepiped.csv')
        self.voxel_centers = self.voxel_data[:, :3]
        self.voxel_indices = self.voxel_data[:, 3:]

        # Translate voxels in front of robot
        y_trans = 0.5
        voxel_centers_shifted = np.copy(self.voxel_centers)
        voxel_centers_shifted[:, 1] += y_trans
        self.voxel_centers = voxel_centers_shifted

        self.paths = np.load('/home/marcus/ros2_ws/src/apple_approach/resources/voxel_paths_parallelepiped.npy')

        self.get_logger().info('Coordinate to trajectory service up and running')

    def coord_to_traj_callback(self, request, response):
        # Extract the requested coordinate
        point_msg = request.coordinate
        x = point_msg.x
        y = point_msg.y
        z = point_msg.z
        point = np.array([x, y, z])

        # Calculate distances
        distances = np.linalg.norm(self.voxel_centers - point, axis=1)

        # Find the index of the closest voxel
        closest_voxel_index = np.argmin(distances)

        self.get_logger().info(f'Distance error: {distances[closest_voxel_index]}')

        # Get the associated path to closest voxel
        path = self.paths[:, :, closest_voxel_index]

        # Convert the NumPy array to a Float32MultiArray
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
        response.waypoints = float32_array
        response.success = True

        if response.success:
            self.waypoint_msg = float32_array
            self.trigger_arm_mover()

        return response

    def trigger_arm_mover(self):
        if not self.client.service_is_ready():
            self.get_logger().info('Waiting for C++ service to be available...')
            self.client.wait_for_service()

        request = SendTrajectory.Request()
        request.waypoints = self.waypoint_msg  # Pass the entire Float32MultiArray message

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Waypoint path service call succeeded')
        else:
            self.get_logger().error('Waypoint path service call failed')

def main():
    rclpy.init()

    coord_to_traj = CoordinateToTrajectoryService()

    rclpy.spin(coord_to_traj)

    rclpy.shutdown()


if __name__ == '__main__':
    main()