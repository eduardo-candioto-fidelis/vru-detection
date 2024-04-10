import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

# Load the map
world = client.get_world()

# Get the map
carla_map = world.get_map()

# Get the waypoints from the map
#waypoints = carla_map.generate_waypoints(distance=2.0)
waypoints = [carla.Location(x=-105, y=4, z=0),
             carla.Location(x=-105, y=-7, z=0),
             carla.Location(x=-100, y=4, z=0),
             carla.Location(x=-100, y=-7, z=0)]

# Create a marker blueprint
#print(world.get_blueprint_library('static'))
marker_bp = world.get_blueprint_library().find('static.prop.box01')

# Iterate through the waypoints and spawn markers at their locations
for waypoint in waypoints:
    #location = waypoint.transform.location
    #print(location)
    world.spawn_actor(marker_bp, carla.Transform(waypoint))
