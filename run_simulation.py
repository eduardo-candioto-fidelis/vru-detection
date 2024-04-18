#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Lidar projection on RGB camera example
"""

import glob
import os
import sys

import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import random
import time
import logging
from queue import Queue
from queue import Empty
from matplotlib import cm

import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')


# Connect to the server
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(2.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(1.0)

original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.3
#settings.fixed_delta_seconds = None
world.apply_settings(settings)

synchronous_master = False

def get_args():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=1,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    argparser.add_argument(
        '--rotation-frequency',
        metavar='N',
        default='10',
        type=float,
        help='Lidar rotation frequency (default: 10.0)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1
    return args    

def spawn_crossing_npcs(args):
    vehicles_list = []
    walkers_list = []
    all_id = []
    
    blueprints = world.get_blueprint_library().filter(args.filterv)
    blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

    if args.safe:
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if args.number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif args.number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
        args.number_of_vehicles = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= args.number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        if args.car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 0.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(args.number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    destination1 = carla.Location(x=17, y=-180, z=0)
    destination2 = carla.Location(x=-8, y=-180, z=0)
    destination_actors = []
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        # all_actors[i].go_to_location(world.get_random_location_from_navigation())
        #print(world.get_random_location_from_navigation())
        current_location = all_actors[i].get_location()
        
        if current_location.x > 0:
            all_actors[i].go_to_location(destination1)
            destination_actors.append(destination1)
        elif current_location.x <= 0:
            all_actors[i].go_to_location(destination2)
            destination_actors.append(destination2)
        destination_actors.append(None)
        #all_actors[i].go_to_location(destination1)
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)

    concluded_ids = []

    return (all_id, 
            concluded_ids,
            all_actors,
            destination1,
            destination2,
            destination_actors,
            vehicles_list,
            walkers_list)


def update_npcs_path(all_id, 
                    concluded_ids,
                    all_actors,
                    destination1,
                    destination2,
                    destination_actors,
                    vehicles_list,
                    walkers_list):
    for i in range(0, len(all_id), 2):
        #import code; code.interact(local=locals())
        if i in concluded_ids:
            continue
        current_location = all_actors[i].get_location()
        #print(current_location.distance(destination))
        actor_destination = destination_actors[i]
        if current_location.distance(actor_destination) <= 3.0:
            #print(all_actors[i])
            if actor_destination == destination1:
                #print(1)
                new_destination = destination2
            elif actor_destination == destination2:
                #print(2)
                new_destination = destination1
            all_actors[i].go_to_location(new_destination)
            concluded_ids.append(i)
        #print(current_location)


def delete_npcs(all_id, 
                concluded_ids,
                all_actors,
                destination1,
                destination2,
                destination_actors,
                vehicles_list,
                walkers_list):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    # stop walker controllers (list is [controller, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        all_actors[i].stop()

    print('\ndestroying %d walkers' % len(walkers_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

    time.sleep(0.5)


def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)


def setup_lidar(args):
    camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
    lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]

    # Configure the blueprints
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))

    if args.no_noise:
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
    lidar_bp.set_attribute('channels', str(args.channels))
    lidar_bp.set_attribute('range', str(args.range))
    lidar_bp.set_attribute('points_per_second', str(args.points_per_second))
    lidar_bp.set_attribute('rotation_frequency', str(args.points_per_second))

    # Spawn the blueprints
    transform = carla.Transform(location=carla.Location(x=-9, y=-178, z=5),
                                rotation=carla.Rotation(pitch=-10, yaw=-50, roll=0))
    camera = world.spawn_actor(
        blueprint=camera_bp,
        transform=transform)
    lidar = world.spawn_actor(
        blueprint=lidar_bp,
        transform=transform)

    # Build the K projection matrix:
    # K = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0

    # The sensor data will be saved in thread-safe Queues
    image_queue = Queue()
    lidar_queue = Queue()

    camera.listen(lambda data: sensor_callback(data, image_queue))
    lidar.listen(lambda data: sensor_callback(data, lidar_queue))   

    VIRIDIS = np.array(cm.get_cmap('viridis').colors)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

    return (image_queue, lidar_queue,
            lidar, camera,
            K,
            image_w, image_h,
            VID_RANGE, VIRIDIS)  
    

def read_lidar(args, frame,
               image_queue, lidar_queue,
               lidar, camera,
               K,
               image_w, image_h,
               VID_RANGE, VIRIDIS):
    if frame > args.frames:
        return
    world_frame = world.get_snapshot().frame

    try:
        # Get the data once it's received.
        image_data = image_queue.get(True, 1.0)
        lidar_data = lidar_queue.get(True, 1.0)
    except Empty:
        print("[Warning] Some sensor data has been missed")

    assert image_data.frame == lidar_data.frame == world_frame
    # At this point, we have the synchronized information from the 2 sensors.
    sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d" %
        (frame, args.frames, world_frame, image_data.frame, lidar_data.frame) + ' ')
    sys.stdout.flush()

    # Get the raw BGRA buffer and convert it to an array of RGB of
    # shape (image_data.height, image_data.width, 3).
    im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
    im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
    im_array = im_array[:, :, :3][:, :, ::-1]

    image = im_array.copy()

    # Get the lidar data and convert it to a numpy array.
    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(p_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
    
    # This (4, 4) matrix transforms the points from lidar space to world space.
    lidar_2_world = lidar.get_transform().get_matrix()

    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarted, the same with points behind the camera projection plane.
    points_2d = points_2d.T
    intensity = intensity.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    points_2d = points_2d[points_in_canvas_mask]
    intensity = intensity[points_in_canvas_mask]

    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[:, 0].astype(np.int)
    v_coord = points_2d[:, 1].astype(np.int)

    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    intensity = 4 * intensity - 3
    color_map = np.array([
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

    if args.dot_extent <= 0:
        # Draw the 2d points on the image as a single pixel using numpy.
        im_array[v_coord, u_coord] = color_map
    else:
        # Draw the 2d points on the image as squares of extent args.dot_extent.
        for i in range(len(points_2d)):
            # I'm not a NumPy expert and I don't know how to set bigger dots
            # without using this loop, so if anyone has a better solution,
            # make sure to update this script. Meanwhile, it's fast enough :)
            im_array[
                v_coord[i]-args.dot_extent : v_coord[i]+args.dot_extent,
                u_coord[i]-args.dot_extent : u_coord[i]+args.dot_extent] = color_map[i]

    return im_array, p_cloud, image


def delete_lidar(args, frame,
                 image_queue, lidar_queue,
                 lidar, camera,
                 K,
                 image_w, image_h,
                 VID_RANGE, VIRIDIS):
    world.apply_settings(original_settings)
    # Destroy the actors in the scene.
    if camera:
        camera.destroy()
    if lidar:
        lidar.destroy()   


def show_frame(im_array):
    cv2.imshow('Frames', cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR ))
    cv2.waitKey(1)
    

def to_lidar_coordinate(loc, w2l):
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to lidar coordinates
    point_lidar = np.dot(w2l, point)

    return point_lidar
    

def get_image_point(point_lidar, K):
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_lidar = [point_lidar[1], -point_lidar[2], point_lidar[0]]

    # now project 3D->2D using the lidar matrix
    point_img = np.dot(K, point_lidar)

    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def draw_bounding_boxes(im_array, pedestrians_bb,
                        image_queue, lidar_queue,
                        lidar, camera,
                        K,
                        image_w, image_h,
                        VID_RANGE, VIRIDIS):
    '''
    Esta função é baseada no tutorial: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/.
    '''
    global edges
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    im_array = im_array.copy().astype('uint8')

    for pedestrian in pedestrians_bb:
        for edge in edges:
            p1 = get_image_point(pedestrian[edge[0]], K)
            p2 = get_image_point(pedestrian[edge[1]],  K)
            cv2.line(im_array, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 1)

    return im_array


def get_bounding_boxes(im_array,
                        image_queue, lidar_queue,
                        lidar, camera,
                        K,
                        image_w, image_h,
                        VID_RANGE, VIRIDIS):
    '''
    Esta função é baseada no tutorial: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/.
    '''
    global edges
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    im_array = im_array.copy().astype('uint8')

    world_2_lidar = np.array(lidar.get_transform().get_inverse_matrix())

    pedestrians_bb = []

    for pedestrian in world.get_actors().filter('walker.pedestrian.*'):
        bb = pedestrian.bounding_box
        dist = pedestrian.get_transform().location.distance(lidar.get_transform().location)

        # Filter for the vehicles within 50m
        if dist < 50:

        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the other vehicle. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = lidar.get_transform().get_forward_vector()
            forward_aray = np.array([[forward_vec.x], [forward_vec.y], [forward_vec.z]])
            ray = pedestrian.get_transform().location - lidar.get_transform().location
            ray_array = np.array([[ray.x], [ray.y], [ray.z]])
            
            if forward_aray.T.dot(ray_array) > 1:
                verts = [v for v in bb.get_world_vertices(pedestrian.get_transform())]
                verts_lidar = [to_lidar_coordinate(v, world_2_lidar) for v in verts]
                pedestrians_bb.append(verts_lidar)

    return pedestrians_bb
    

def create_dataset_dir(name):
    os.mkdir(f'./{name}')
    os.mkdir(f'./{name}/images/')
    os.mkdir(f'./{name}/point_clouds/')
    os.mkdir(f'./{name}/bounding_boxes/')


def store_in_dataset(frame, name, image, p_cloud, pedestrians_bb):
    cv2.imwrite(f'./{name}/images/img-{frame:05d}.png', image)
    np.save(f'./{name}/point_clouds/pc-{frame:05d}.npy', p_cloud)
    np.save(f'./{name}/bounding_boxes/bb-{frame:05d}.npy', pedestrians_bb)
    

def main():
    name = 'dataset'
    create_dataset_dir(name)
    args = get_args()
    npcs_info = spawn_crossing_npcs(args)
    lidar_info = setup_lidar(args)
    try:
        frame = 0
        while True:
            world.tick()
            im_array, p_cloud, image = read_lidar(args, frame, *lidar_info)
            update_npcs_path(*npcs_info)
            if im_array is not None:
                pedestrians_bb = get_bounding_boxes(im_array, *lidar_info)
                im_array = draw_bounding_boxes(im_array, pedestrians_bb, *lidar_info)
                show_frame(im_array)
                store_in_dataset(frame, name, image, p_cloud, pedestrians_bb)
            frame += 1
    finally:
        delete_lidar(args, frame, *lidar_info)
        delete_npcs(*npcs_info)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')