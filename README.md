# Generate PRECOG Data

## Screenshot of run

![run screenshot](https://raw.githubusercontent.com/fireofearth/generate_precog_data/master/assets/precog_generate_data.png)

## Plot of sample

![plot sample](https://raw.githubusercontent.com/fireofearth/generate_precog_data/master/assets/plot_sample.png)

## Synopsis

Collects data for PRECOG ESP model presented in <https://people.eecs.berkeley.edu/~rmcallister/publication/precog/> for *CARLA 0.9*

For each map in the CARLA simulator, I run 4 episodes of data collection. For each episodes, I spawn 80 cars and add LIDAR sensors randomly to 20 of them. These vehicles roam the map on autopilot for 1000 frames, and the cars with sensors capture samples at 10 frame intervals (not capturing the samples during the first 50 frames as cars just spawning are accelerating from rest). After capturing, I only save the samples that are in the scenarios of {uncontrolled intersections, no intersections} (or just {uncontrolled intersections} if a map has few uncontrolled intersections), augmenting the samples by random rotation about the origin.

## Installation and usage.

1. Install CARLA version 0.9.X where X>=11 from Github releases <https://github.com/carla-simulator/carla/releases>.
2. Create a Python environment using `venv` or Anaconda, etc.
3. `pip install -r generate_precog_data/requirements.txt`.
4. Launch CARLA by running `carla/CarlaUE4.sh` in your CARLA installation path.
5. Git clone the PRECOG repository `https://github.com/fireofearth/precog` (no need to install deps for PRECOG).
5. Double check the Bash exported paths in `generate_precog_data/env.sh` for PRECOG and CARLA are correct.
6. `source generate_precog_data/env.sh`.
7. Run the example collect data script `generate_precog_data/collect_data.sh`.

```
usage: run.py [-h] [-v] [--host H] [-p P] [--dir SAVE_DIRECTORY] [-s SEED]
              [--map MAP] [-e E] [-f F] [-b B] [-n N] [-d D] [--augment-data]
              [--n-augments N_AUGMENTS] [--hybrid] [--car-lights-on]

CARLA Automatic Control Client

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Show debug information
  --host H              IP of the host server (default: 127.0.0.1)
  -p P, --port P        TCP port to listen to (default: 2000)
  --dir SAVE_DIRECTORY  Directory to save the samples.
  -s SEED, --seed SEED  Set seed for repeating executions (default: None)
  --map MAP             Set the CARLA map to collect data from.
  -e E, --n-episodes E  Number of episodes to run (default: 10)
  -f F, --n-frames F    Number of frames in each episode to capture (default:
                        1000)
  -b B, --n-burn-frames B
                        Number of frames at the beginning of each episode to
                        skip data collection (default: 60)
  -n N, --n-vehicles N  number of vehicles (default: 80)
  -d D, --n-data-collectors D
                        number of data collectos to add on vehicles (default:
                        20)
  --augment-data        Enable data augmentation
  --n-augments N_AUGMENTS
                        Number of augmentations to create from each sample. If
                        --n-aguments=5 then a random number from 1 to 5
                        augmentations will be produced from each sample
  --hybrid              Enanble
  --car-lights-on       Enable car lights
```
