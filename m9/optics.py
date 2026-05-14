import numpy as np
from dataclasses import dataclass


EPS = 1e-12


@dataclass
class Ray:
    x: float
    y: float
    theta: float
    power: float = 1.0
    alive: bool = True
    path: list | None = None

    def __post_init__(self):
        if self.path is None:
            self.path = [(self.x, self.y)]

    @property
    def direction(self):
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=float)

    def propagate_distance(self, distance):
        if not self.alive:
            return
        direction = self.direction
        self.x += distance * direction[0]
        self.y += distance * direction[1]
        self.path.append((self.x, self.y))

    def propagate_to_x(self, x_target):
        if not self.alive:
            return
        direction = self.direction
        if abs(direction[0]) < EPS:
            self.alive = False
            return
        t = (x_target - self.x) / direction[0]
        if t < 0:
            self.alive = False
            return
        self.propagate_distance(t)


class ThinLens:
    def __init__(self, x, f, aperture_radius):
        self.x = x
        self.f = f
        self.aperture_radius = aperture_radius
        self.entrance_x = x
        self.exit_x = x

    def interact(self, ray):
        ray.propagate_to_x(self.x)
        if not ray.alive:
            return ray
        if abs(ray.y) > self.aperture_radius:
            ray.alive = False
            return ray
        slope = np.tan(ray.theta)
        slope_out = slope - ray.y / self.f
        ray.theta = np.arctan(slope_out)
        return ray


class OpticalSystem:
    def __init__(self, elements, image_plane_x):
        self.elements = elements
        self.image_plane_x = image_plane_x

    @property
    def first_element(self):
        return self.elements[0]

    @property
    def last_exit_x(self):
        return self.elements[-1].exit_x

    def trace_to_exit(self, ray):
        for element in self.elements:
            element.interact(ray)
            if not ray.alive:
                return ray
        return ray

    def trace_to_image_plane(self, ray):
        self.trace_to_exit(ray)
        if not ray.alive:
            return ray
        ray.propagate_to_x(self.image_plane_x)
        return ray
