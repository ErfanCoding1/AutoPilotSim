import os
import time
import sys
import random

sys.path.append(
    'D:\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' % (sys.version_info.major,
                                                                                         sys.version_info.minor))

import carla



# ------------------ Helper Functions ------------------
def connect_to_carla(host='localhost', port=2000, timeout=10.0):
    """
    Connect to the Carla server and retrieve the simulation world.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def create_vehicle(world, spawn_point_index=0):
    """
    Spawn a vehicle from the blueprint library at a selected spawn point.
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_point_index])
    vehicle.set_autopilot(True)
    return vehicle


def create_camera(world, vehicle, location=carla.Location(x=1.5, z=2.4), rotation=carla.Rotation()):
    """
    Create an RGB camera sensor and attach it to the vehicle.
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(location, rotation)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera


def set_weather(world, weather_parameters):
    """
    Set the simulation world's weather to the specified parameters.
    """
    world.set_weather(weather_parameters)


def initialize_dataset_directory(root_dir='carla_weather_dataset'):
    """
    Create the root dataset directory along with two subdirectories: train and test.
    """
    os.makedirs(root_dir, exist_ok=True)


def save_image(image, base_dir, weather_name, image_count):
    """
    Save an image to disk in either the train or test folder, based on random assignment.
    The image is saved directly in the train or test folder (without creating a weather-specific subfolder).
    File name includes the weather name.
    """
    os.makedirs(base_dir, exist_ok=True)
    image_filename = os.path.join(base_dir, f'{weather_name}_{image_count:06d}.png')
    image.save_to_disk(image_filename)



def turn_off_traffic_lights(world):
    """
    Disable (turn off) all traffic lights in the simulation so that red lights do not force vehicles to stop.
    """
    actors = world.get_actors()
    traffic_lights = [actor for actor in actors if 'traffic_light' in actor.type_id]
    for tl in traffic_lights:
        tl.set_state(carla.TrafficLightState.Off)
        tl.freeze(True)
    print("All traffic lights have been disabled.")


def capture_images_for_weather(camera, weather_name, duration=10, output_dir='carla_weather_dataset'):
    """
    Collect images for a given weather condition for a specific duration.
    The callback function saves each frame into the designated train or test folder.
    """
    image_count = 0

    def image_callback(image):
        nonlocal image_count
        save_image(image, output_dir, weather_name, image_count)
        image_count += 1

    camera.listen(image_callback)
    print(f"Starting image capture for weather: {weather_name} for {duration} seconds...")
    time.sleep(duration)
    camera.stop()
    print(f"Captured {image_count} images for weather: {weather_name}")
    return image_count


def cleanup_actors(actors):
    """
    Destroy all actors to release resources.
    """
    for actor in actors:
        if actor is not None:
            actor.destroy()


# ------------------ Main Function ------------------
def main():
    # Define custom weather parameters for all weather conditions

    clear_noon = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        sun_altitude_angle=75.0,
        sun_azimuth_angle=0.0,
        fog_density=0.0,
        fog_distance=100.0,
        fog_falloff=1.0,
        wetness=0.0
    )

    clear_night = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        sun_altitude_angle=-30.0,
        sun_azimuth_angle=0.0,
        fog_density=0.0,
        fog_distance=100.0,
        fog_falloff=1.0,
        wetness=0.0
    )

    wet_cloudy_noon = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        sun_altitude_angle=65.0,
        sun_azimuth_angle=0.0,
        fog_density=0.0,
        fog_distance=100.0,
        fog_falloff=1.0,
        wetness=30.0,
    )

    soft_rain_noon = carla.WeatherParameters(
        cloudiness=70.0,
        precipitation=30.0,
        sun_altitude_angle=60.0,
        sun_azimuth_angle=0.0,
        fog_density=0.0,
        fog_distance=100.0,
        fog_falloff=1.0,
        wetness=70.0,
    )

    foggy = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        sun_altitude_angle=60.0,
        sun_azimuth_angle=0.0,
        fog_density=80.0,
        fog_distance=50.0,
        fog_falloff=2.0,
        wetness=0.0
    )

    weather_presets = {
        'ClearNoon': clear_noon,
        'ClearNight': clear_night,
        'WetCloudyNoon': wet_cloudy_noon,
        'SoftRainNoon': soft_rain_noon,
        'Foggy': foggy,
    }

    # Create dataset base directory (with "train" and "test" subfolders)
    output_dir = 'carla_weather_dataset'
    initialize_dataset_directory(output_dir)

    # Connect to Carla
    client, world = connect_to_carla()

    # Turn off traffic lights (disable red lights)
    turn_off_traffic_lights(world)

    # Spawn vehicle and attach a camera sensor
    vehicle = create_vehicle(world)
    camera = create_camera(world, vehicle)

    collected_images = {}

    # Iterate over each weather condition and capture images.
    for weather_name, weather_param in weather_presets.items():
        print(f"Setting weather to: {weather_name}")
        set_weather(world, weather_param)
        time.sleep(2)  # Allow weather to stabilize
        image_count = capture_images_for_weather(camera, weather_name, duration=10, output_dir=output_dir)
        collected_images[weather_name] = image_count

    # Display summary of image collection
    print("Image collection summary:")
    for weather, count in collected_images.items():
        print(f"{weather}: {count} images")

    # Clean up actors (vehicle, camera)
    cleanup_actors([camera, vehicle])
    print("Dataset collection completed.")


if __name__ == '__main__':
    main()
