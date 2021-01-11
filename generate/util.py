import json
import itertools
import functools
import numpy as np
import networkx as nx
import os
import carla

class NumpyEncoder(json.JSONEncoder):
    """The encoding object used to serialize np.ndarrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_datum(datum, directory, filename):
    """Save datum

    Parameters
    ----------
    datum : dict
        The data to save
    directory : str
        The directory name to save the data.
    filename : str
        The filename to name the data
    """
    assert(os.path.isdir(directory))
    filename = filename if filename.endswith('.json') else f"{filename}.json"
    with open(os.path.join(directory, filename), 'w') as f:
            json.dump(datum, f, cls=NumpyEncoder)

def merge_list_of_list(ll):
    """Concatenate iterable of iterables into one list."""
    return list(itertools.chain.from_iterable(ll))

def map_to_list(f, l):
    """Does map operation and then converts map object to a list."""
    return list(map(f, l))

def filter_to_list(f, l):
    return list(filter(f, l))

def compress_to_list(l, bl):
    return list(itertools.compress(l, bl))

def reduce(f, l, i=None):
    """
    Parameters
    ----------
    f : (function v, acc: f(v, acc))
    l : iterable
    i : any
    """
    return functools.reduce(f, l, i)

def map_to_ndarray(f, l):
    """Does map operation and then converts map object to a list."""
    return np.array(map_to_list(f, l))

def location_to_ndarray(l):
    """Converts carla.Location to ndarray [x, y, z]"""
    return np.array([l.x, l.y, l.z])

def ndarray_to_location(v):
    """ndarray of form [x, y, z] to carla.Location."""
    return carla.Location(x=v[0], y=v[1], z=v[2])

def actor_to_location_ndarray(a):
    """Converts carla.Actor's location ndarray [x, y, z]"""
    return location_to_ndarray(a.get_location())

def actors_to_location_ndarray(alist):
    """Converts iterable of carla.Actor to a ndarray of size (len(iterable), 3)"""
    return np.array(map_to_list(actor_to_location_ndarray, alist))

def transform_to_location_ndarray(t):
    """Converts carla.Transform to location ndarray [x, y, z]"""
    return location_to_ndarray(t.location)

def transforms_to_location_ndarray(ts):
    """Converts an iterable of carla.Transform to a ndarray of size (len(iterable), 3)"""
    return np.array(map_to_list(transform_to_location_ndarray, ts))

def transform_points(transform, points):
    """Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]."""
    # Needed format: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    # points = transform * points
    points = np.dot(transform, points)
    # Return all but last row
    return points[0:3].transpose()

def transform_to_origin(transform, origin):
    """Create an adjusted transformation relative to origin.
    Creates a new transformation (doesn't mutate the parameters).
    
    Parameters
    ----------
    transform : carla.Transform
        The transform we want to adjust
    origin : carla.Transform or np.array
        The origin we want to adjust the transform to

    Returns
    -------
    carla.Transform
        New transform with the origin as reference.
    """
    location = transform.location
    rotation = transform.rotation
    return carla.Transform(
            carla.Location(
                x=location.x - origin.location.x,
                y=location.y - origin.location.y,
                z=location.z - origin.location.z),
            carla.Rotation(
                pitch=rotation.pitch,
                yaw=rotation.yaw - origin.rotation.yaw,
                roll=rotation.roll))

def get_junctions_from_topology_graph(topology):
    """Gets unique junctions from topology

    Parameters
    ----------
    topology : nx.Graph or list

    Returns
    -------
    list of carla.Junction
    """
    if isinstance(topology, list):
        G = nx.Graph()
        G.add_edges_from(topology)
        topology = G
    junctions = map(lambda v: v.get_junction(),
        filter(lambda v: v.is_junction, topology.nodes))
    return list({j.id: j for j in junctions}.values())
