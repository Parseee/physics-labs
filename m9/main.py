import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Ray:
    def __init__(self, y, theta):
        self.y = y        # Height [m]
        self.theta = theta  # Angle [rad]


class Surface:
    def __init__(self, R, n1, n2):
        self.R = R       # Radius of curvature
        self.n1 = n1     # Index before
        self.n2 = n2     # Index after

    def refract(self, ray):
        # Exact Snell's law: n1*sin(i) = n2*sin(t)
        # i = theta + alpha_normal, alpha_normal = -arcsin(y/R)
        if self.R == 0:
            return ray  # Flat surface

        alpha = -np.arcsin(np.clip(ray.y / self.R, -1, 1))
        theta_i = ray.theta - alpha

        # Snell's Law
        sin_t = (self.n1 / self.n2) * np.sin(theta_i)
        theta_t = np.arcsin(np.clip(sin_t, -1, 1))

        ray.theta = theta_t + alpha
        return ray


class ThickLens:
    def __init__(self, R1, R2, d, n):
        self.s1 = Surface(R1, 1.0, n)
        self.s2 = Surface(R2, n, 1.0)
        self.d = d  # Thickness

    def trace(self, ray):
        # Propagate to s1, refract, prop d, refract s2
        ray = self.s1.refract(ray)
        ray.y += ray.theta * self.d
        ray = self.s2.refract(ray)
        return ray


def get_path(ray_start, lens, x_f):
    # ray_start = (x, y, theta)
    x, y, theta = ray_start

    # 1. Propagate to Surface 1 (at -d/2)
    dist1 = -lens.d/2 - x
    y1 = y + dist1 * np.tan(theta)

    # 2. Refract Surface 1
    ray_temp = Ray(y1, theta)
    ray_temp = lens.s1.refract(ray_temp)

    # 3. Propagate inside (to d/2)
    dist2 = lens.d
    y2 = y1 + dist2 * np.tan(ray_temp.theta)

    # 4. Refract Surface 2
    ray_temp = lens.s2.refract(ray_temp)

    # 5. Propagate to focal plane
    dist3 = x_f - (lens.d/2)
    y3 = y2 + dist3 * np.tan(ray_temp.theta)

    return [(-0.5, y), (-lens.d/2, y1), (lens.d/2, y2), (x_f, y3)]

# Plotting


# Parameters
f = 0.1  # Focal length
R = 0.1 * 0.5  # Radius approx
d = 0.05
H = d * 5  # lens visual height
lens = ThickLens(R, -R, d, 1.5)  # Thick lens
object_h = 0.02
rays = [Ray(y, 0) for y in np.linspace(-object_h, object_h, 10)]


# Plotting
fig, ax = plt.subplots()
for r in rays:
    path = get_path((-0.5, r.y, r.theta), lens, 0.15)
    xs, ys = zip(*path)
    plt.plot(xs, ys, 'b-')
ax.set_title("Ray Tracing: Thick Lens")

lens_rect = Rectangle((-d/2, -H/2), d, H, color='blue', alpha=0.3)
ax.add_patch(lens_rect)
plt.show()
