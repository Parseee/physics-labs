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


def launch_angles_for_aperture(x_obj, y_obj, element, n_angles):
    a1 = np.arctan2(-element.aperture_radius - y_obj, element.entrance_x - x_obj)
    a2 = np.arctan2(+element.aperture_radius - y_obj, element.entrance_x - x_obj)
    if a2 < a1:
        a1, a2 = a2, a1
    return np.linspace(a1, a2, n_angles)


def build_object_profile(y_grid):
    profile = np.full_like(y_grid, 0.03, dtype=float)
    profile += 0.92 * (np.abs(y_grid) < 0.0007)
    profile += 0.70 * ((y_grid > 0.004) & (y_grid < 0.006))
    profile += 0.70 * ((y_grid > -0.006) & (y_grid < -0.004))
    profile += 0.55 * ((y_grid > 0.009) & (y_grid < 0.010))
    profile += 0.55 * ((y_grid > -0.010) & (y_grid < -0.009))
    return profile / profile.max()


def deposit_to_detector(detector_y, detector_signal, y_hit, weight):
    if y_hit < detector_y[0] or y_hit > detector_y[-1]:
        return
    idx = np.searchsorted(detector_y, y_hit)
    if idx <= 0:
        detector_signal[0] += weight
        return
    if idx >= len(detector_y):
        detector_signal[-1] += weight
        return
    y0 = detector_y[idx - 1]
    y1 = detector_y[idx]
    if abs(y1 - y0) < EPS:
        detector_signal[idx] += weight
        return
    t = (y_hit - y0) / (y1 - y0)
    detector_signal[idx - 1] += weight * (1.0 - t)
    detector_signal[idx] += weight * t


def render_line_object(system, x_object, object_y, object_intensity, detector_y, n_angles=241):
    detector_signal = np.zeros_like(detector_y, dtype=float)
    first = system.first_element
    for y_obj, intensity in zip(object_y, object_intensity):
        if intensity <= 0:
            continue
        angles = launch_angles_for_aperture(x_object, y_obj, first, n_angles)
        d_alpha = abs(angles[1] - angles[0]) if len(angles) > 1 else 1.0
        for alpha in angles:
            ray = Ray(x=x_object, y=y_obj, theta=alpha, power=float(intensity * d_alpha))
            system.trace_to_image_plane(ray)
            if ray.alive:
                deposit_to_detector(detector_y, detector_signal, ray.y, ray.power)
    max_val = detector_signal.max()
    if max_val > 0:
        detector_signal /= max_val
    return detector_signal


def point_response_stats(system, x_object, y_object, n_angles=401):
    first = system.first_element
    angles = launch_angles_for_aperture(x_object, y_object, first, n_angles)
    y_hits = []
    weights = []
    for alpha in angles:
        ray = Ray(x=x_object, y=y_object, theta=alpha, power=1.0)
        system.trace_to_image_plane(ray)
        if ray.alive:
            y_hits.append(ray.y)
            weights.append(ray.power)
    if not y_hits:
        return np.nan, np.nan, 0.0
    y_hits = np.array(y_hits)
    weights = np.array(weights)
    w = np.maximum(weights, 0.0)
    if np.sum(w) < EPS:
        return np.nan, np.nan, 0.0
    centroid = np.sum(w * y_hits) / np.sum(w)
    rms = np.sqrt(np.sum(w * (y_hits - centroid) ** 2) / np.sum(w))
    return centroid, rms, np.sum(w)


def paraxial_image_point(system, x_object, y_object, launch_angles=(-2e-3, 2e-3)):
    rays = []
    for alpha in launch_angles:
        ray = Ray(x=x_object, y=y_object, theta=alpha, power=1.0)
        system.trace_to_exit(ray)
        if not ray.alive:
            return np.nan, np.nan
        rays.append(ray)
    r1, r2 = rays
    m1 = np.tan(r1.theta)
    m2 = np.tan(r2.theta)
    if abs(m1 - m2) < EPS:
        return np.nan, np.nan
    x_img = (r2.y - r1.y + m1 * r1.x - m2 * r2.x) / (m1 - m2)
    y_img = r1.y + m1 * (x_img - r1.x)
    return x_img, y_img
