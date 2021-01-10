#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import time
import numpy as np

sys.path.append(
    os.path.join(
        os.getenv('CARLA_DIR'),
        'PythonAPI/carla'))

import carla
from carla import VehicleLightState as vls
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor

import generate.util as util
from generate.data import DataCollector, IntersectionReader

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        # n_frames : int
        #     Number of frames to collect data in each episode.
        #     Note: a vehicle takes roughly 130 frames to make a turn.
        self.n_frames = 130 * 20
        # delta : float
        #     Step size for synchronous mode.
        self.delta = 0.1
        if self.args.seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.args.seed)

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)        
        self.intersection_reader = IntersectionReader(
                self.world, self.carla_map)

    def setup_players(self):
        """
        Parameters
        ----------
        client : carla.Client
        carla_map : carla.Map
        args : argparse.Namespace

        Returns
        -------
        list of int
            IDs of 4 wheeled vehicles on autopilot.
        list of DataCollector
            Data collectors with set up done and listening to LIDAR.
        """
        vehicle_ids = []
        data_collectors = []

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # if args.safe:
        #     blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        #     blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.carla_map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.args.number_of_vehicles < number_of_spawn_points:
            np.random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points

        # Generate vehicles
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if self.args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        # Wait for vehicles to finish generating
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicle_ids.append(response.actor_id)
        
        # Add data collector to a handful of vehicles
        vehicles_ids_to_data_collect = vehicle_ids[
                :self.args.number_of_data_collectors]
        for idx, vehicle_id in enumerate(vehicles_ids_to_data_collect):
            vehicle_ids_to_watch = vehicle_ids[:idx] + vehicle_ids[idx + 1:]
            vehicle = self.world.get_actor(vehicle_id)
            data_collector = DataCollector(vehicle,
                    intersection_reader=self.intersection_reader)
            data_collector.start_sensor()
            data_collector.set_vehicles(vehicle_ids_to_watch)
            data_collectors.append(data_collector)

        logging.info(f"spawned {len(vehicle_ids)} vehicles")
        return vehicle_ids, data_collectors

    def run(self):
        """ Main loop for agent"""
        vehicle_ids = []
        data_collectors = []
        world = None
        original_settings = None

        try:
            logging.info("Turning on synchronous setting and updating traffic manager.")
            original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = self.delta
            settings.synchronous_mode = True

            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            self.traffic_manager.global_percentage_speed_difference(0.0)
            if self.args.seed is not None:
                self.traffic_manager.set_random_device_seed(self.args.seed)        
            if self.args.hybrid:
                self.traffic_manager.set_hybrid_physics_mode(True)
            self.world.apply_settings(settings)

            logging.info("Create vehicles and data collectors.")
            vehicle_ids, data_collectors = self.setup_players()
            data_collector = data_collectors[0]
            logging.info("Running simulation.")
            for idx in range(self.n_frames):
                frame = self.world.tick()
                for data_collector in data_collectors:
                    data_collector.capture_step(frame)

        finally:
            logging.info("Destroying data collectors.")
            if data_collectors:
                for data_collector in data_collectors:
                    data_collector.destroy()
            
            logging.info("Destroying vehicles.")
            if vehicle_ids:
                self.client.apply_batch(
                        [carla.command.DestroyActor(x) for x in vehicle_ids])

            logging.info("Reverting to original settings.")
            if original_settings:
                self.world.apply_settings(original_settings)


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        # default='vehicle.*',
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=80,
        type=int,
        help='number of vehicles (default: 80)')
    argparser.add_argument(
        '-d', '--number-of-data-collectors',
        metavar='D',
        default=5,
        type=int,
        help='number of data collectos to add on vehicles (default: 20)')
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
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')

    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        generator = DataGenerator(args)
        generator.run()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
