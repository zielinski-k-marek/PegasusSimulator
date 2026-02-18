"""
| File: monocular_camera.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Simulates a monocular camera attached to the vehicle
"""
__all__ = ["MonocularCamera"]

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from isaacsim.sensors.camera.camera import Camera
from omni.usd import get_stage_next_free_path

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation


class MonocularCamera(GraphicalSensor):
    """
    The class that implements a monocular camera sensor. This class inherits the base class GraphicalSensor.
    """

    def __init__(self, camera_name, config={}):
        """
        Initialize the MonocularCamera class
        
        Check the oficial documentation for the Camera class in Isaac Sim: 
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#isaac-sim-sensors-camera

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the MonocularCamera - it can be empty or only have some of the parameters used by the MonocularCamera.

        Examples:
            The dictionary default parameters are

            >>> {"depth": True,
            >>> "position": np.array([0.30, 0.0, 0.0]),
            >>> "orientation": np.array([0.0, 0.0, 0.0]),
            >>> "resolution": (1920, 1200),
            >>> "frequency": 30,
            >>> "intrinsics": np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]),
            >>> "distortion_coefficients": [0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]),
            >>> "diagonal_fov": 140.0}
        """

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="MonocularCamera", update_rate=config.get("frequency", 60.0))        
        
        # Setup the name of the camera primitive path
        self._camera_name = camera_name
        self._stage_prim_path = ""

        # Configurations of the camera
        self._depth = config.get("depth", True)
        self._position = config.get("position", np.array([0.30, 0.0, 0.0]))
        self._orientation = config.get("orientation", np.array([0.0, 0.0, 180.0]))
        self._resolution = config.get("resolution", (1920, 1200))
        self._frequency = config.get("frequency", 30)
        self._intrinsics = config.get("intrinsics", np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]))
        self._distortion_coefficients = config.get("distortion_coefficients",[0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017])
        self._diagonal_fov = config.get("diagonal_fov", 140.0)
        self._clipping_range = config.get("clipping_range", (0.01, 100.0))

        # Setup an empty camera output dictionary
        self._state = {}
        self._camera_full_set = False

        self.counter = 0


    def initialize(self, vehicle):
        
        # Initialize the Super class "object" attributes
        super().initialize(vehicle)

        # Get the complete stage prefix for the camera
        self._stage_prim_path = get_stage_next_free_path(PegasusInterface().world.stage, self._vehicle.prim_path + "/body/" + self._camera_name, False)

        # Get the camera name that was actually created (and update the camera name)
        self._camera_name = self._stage_prim_path.rpartition("/")[-1]

        # Create the camera object attached to the rigid body vehicle
        self._camera = Camera(
            prim_path=self._stage_prim_path,
            frequency=self._frequency,
            resolution=self._resolution)
        
        # Set the camera position locally with respect to the drone
        # QUATERNION BUG: scipy as_quat() returns [x,y,z,w] but Isaac Sim set_local_pose
        # expects [w,x,y,z]. Values are passed as-is (no reordering), so Isaac Sim
        # misinterprets the quaternion. Camera default look direction is +X in camera space.
        # Combined with the bug, empirically verified orientations (ZYX Euler):
        #   [0, -90, 0] → downward-looking   [0, 0, 0] → backward-looking
        #   [0, 90, 0]  → upward-looking      [0, 0, 180] → forward-looking
        # Position must clear drone geometry (Iris: use z <= -0.25 to avoid landing gear)
        self._camera.set_local_pose(np.array(self._position), Rotation.from_euler("ZYX", self._orientation, degrees=True).as_quat())
        
    def start(self):

        # Set the camera intrinsics
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = self._intrinsics

        # Start the camera
        self._camera.initialize()

        # Set the camera FOV via focal length + aperture on the USD prim.
        # set_rational_polynomial_properties() only configures the distortion model —
        # it does NOT set the underlying camera's FOV.  We must set focal_length and
        # horizontal_aperture directly so Isaac Sim renders with the correct FOV.
        import math
        _diag_px = math.sqrt(self._resolution[0]**2 + self._resolution[1]**2)
        _f_px = (_diag_px / 2.0) / math.tan(math.radians(self._diagonal_fov / 2.0))
        # Use a reference sensor width (20.955 mm is Isaac Sim's default horizontal aperture)
        _sensor_w = 20.955
        _f_mm = _f_px * _sensor_w / self._resolution[0]
        self._camera.set_focal_length(_f_mm)
        self._camera.set_horizontal_aperture(_sensor_w)

        # Set the correct properties of the camera (this must be done after the camera object is initialized)
        self._camera.set_lens_distortion_model("OmniLensDistortionOpenCvPinholeAPI")
        self._camera.set_rational_polynomial_properties(nominal_width=self._resolution[0], nominal_height=self._resolution[1], optical_centre_x=cx, optical_centre_y=cy, max_fov=self._diagonal_fov, distortion_model=self._distortion_coefficients)
        self._camera.set_clipping_range(*self._clipping_range)

        # Check if depth is enabled, if so, set the depth properties
        if self._depth:
            self._camera.add_distance_to_image_plane_to_frame()

        # Signal that the camera is fully set
        self._camera_full_set = True

        # Diagnostic: verify render product was created
        rp = getattr(self._camera, '_render_product_path', None) or getattr(self._camera, 'render_product_path', None)
        print(f"[MonocularCamera] Camera initialized: prim={self._stage_prim_path} "
              f"res={self._resolution} fov={self._diagonal_fov}° "
              f"render_product={rp}", flush=True)

    def stop(self):
        self._camera_full_set = False

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state


    @GraphicalSensor.update_at_rate
    def update(self, state: State, dt: float):
        """Method that gets the current RGB image from the camera and returns it as a dictionary.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """

        while self.counter < 100:
            self.counter += 1
            return

        # If all the camera properties are not set yet, return None
        if not self._camera_full_set:
            return None

        # Get the data from the camera
        # TODO: Fix this feature later
        try:
            self._state = {}
            self._state["camera_name"] = self._camera_name
            self._state["stage_prim_path"] = self._stage_prim_path
            rgba = self._camera.get_rgba()
            rgb = rgba[:, :, :3]
            self._state["image"] = rgb
            self._state["height"] = self._resolution[1]
            self._state["width"] = self._resolution[0]
            self._state["frequency"] = self._frequency
            self._state["camera"] = self._camera

            # Diagnostic: check first 5 valid frames from camera
            if self.counter < 110:
                import numpy as _np
                mean_val = float(_np.mean(rgb))
                max_val = float(_np.max(rgb))
                print(f"[MonocularCamera] Frame {self.counter}: "
                      f"rgba={rgba.shape} rgb={rgb.shape} dtype={rgba.dtype} "
                      f"mean={mean_val:.1f} max={max_val:.0f} "
                      f"alpha_mean={float(_np.mean(rgba[:,:,3])):.1f} "
                      f"{'BLACK' if max_val == 0 else 'OK'}", flush=True)

            # Check if we want to get the depth image
            #if self._depth:
            #    self._state["depth"] = self._camera.get_depth()

            if self._camera.get_lens_distortion_model() == "pinhole":
                self._state["intrinsics"] = self._camera.get_intrinsics_matrix()
            
        # If something goes wrong during the data acquisition, just return None
        except:
            self._state = None

        return self._state
