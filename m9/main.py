import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json
from pathlib import Path as FsPath


def thin_lens_geometry(f, lens_x, x_object):
    s = lens_x - x_object
    s_img = 1.0 / (1.0 / f - 1.0 / s)
    m = -s_img / s
    return s, s_img, m


def build_triangle(height=0.010, width=0.008):
    return np.array(
        [
            [-height / 2.0, -width / 2.0],
            [-height / 2.0, +width / 2.0],
            [+height / 2.0, 0.0],
        ],
        dtype=float,
    )


def rasterize_triangle(vertices_yz, y_grid, z_grid):
    yy, zz = np.meshgrid(y_grid, z_grid, indexing="ij")
    points = np.column_stack([yy.ravel(), zz.ravel()])
    mask = Path(vertices_yz).contains_points(points).reshape(len(y_grid), len(z_grid))
    image = np.zeros_like(mask, dtype=float)
    image[mask] = 1.0
    return image


def trace_sample_rays(ax, x_object, y_sources, lens_x, f, x_image, object_height, magnification):
    for y0 in y_sources:
        # Ray 1: parallel to axis -> through back focus
        x1 = [x_object, lens_x, x_image]
        y1 = [y0, y0, y0 - (x_image - lens_x) * y0 / f]
        ax.plot(x1, y1, "tab:blue", lw=1.2)

        # Ray 2: through lens center (undeviated in thin-lens model)
        y_img = y0 * (-(x_image - lens_x) / (lens_x - x_object))
        x2 = [x_object, lens_x, x_image]
        y2 = [y0, 0.0, y_img]
        ax.plot(x2, y2, "tab:orange", lw=1.2)

    image_height = object_height * magnification
    ax.plot([x_object, x_object], [-object_height / 2.0, object_height / 2.0], "g-", lw=3, label="Объект")
    ax.plot(
        [x_image, x_image],
        [-image_height / 2.0, image_height / 2.0],
        "r-",
        lw=3,
        label="Проекция объекта",
    )
    ax.axvline(lens_x, color="k", ls="--", lw=1.2, label="Линза")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_title("Ход лучей")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)


def load_config(path="config.json"):
    defaults = {
        "lens": {
            "focal_length": 0.060,
            "x_position": 0.0,
        },
        "object": {
            "x_position": -0.045,
            "triangle_height": 0.010,
            "triangle_width": 0.008,
        },
        "plot": {
            "y_range": 0.012,
            "z_range": 0.012,
            "grid_points": 500,
            "ray_y_min": -0.004,
            "ray_y_max": 0.004,
            "ray_count": 5,
            "fig_width_in": 14.0,
            "fig_height_in": 4.5,
        },
    }

    cfg_path = FsPath(path)
    if not cfg_path.exists():
        return defaults

    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    for top_key, section in defaults.items():
        if top_key in user_cfg and isinstance(user_cfg[top_key], dict):
            section.update(user_cfg[top_key])
    return defaults


def main():
    cfg = load_config("config.json")
    f = cfg["lens"]["focal_length"]
    lens_x = cfg["lens"]["x_position"]
    x_object = cfg["object"]["x_position"]
    s, s_img, m = thin_lens_geometry(f, lens_x, x_object)
    x_image = lens_x + s_img

    print("=== Тонкая линза ===")
    print(f"f = {f*1000:.1f} мм, s = {s*1000:.1f} мм, s' = {s_img*1000:.1f} мм, m = {m:.3f}")
    print(f"Объект на графике лучей: x = {x_object:.4f} м")
    print(f"Проекция объекта на графике лучей: x = {x_image:.4f} м")

    obj_vertices = build_triangle(
        height=cfg["object"]["triangle_height"],
        width=cfg["object"]["triangle_width"],
    )
    img_vertices = m * obj_vertices

    y_range = cfg["plot"]["y_range"]
    z_range = cfg["plot"]["z_range"]
    n_grid = int(cfg["plot"]["grid_points"])
    y_grid = np.linspace(-y_range, y_range, n_grid)
    z_grid = np.linspace(-z_range, z_range, n_grid)
    object_img = rasterize_triangle(obj_vertices, y_grid, z_grid)
    image_img = rasterize_triangle(img_vertices, y_grid, z_grid)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(cfg["plot"]["fig_width_in"], cfg["plot"]["fig_height_in"]),
    )

    trace_sample_rays(
        axes[0],
        x_object=x_object,
        y_sources=np.linspace(
            cfg["plot"]["ray_y_min"],
            cfg["plot"]["ray_y_max"],
            int(cfg["plot"]["ray_count"]),
        ),
        lens_x=lens_x,
        f=f,
        x_image=x_image,
        object_height=cfg["object"]["triangle_height"],
        magnification=m,
    )

    extent = [z_grid[0] * 1000, z_grid[-1] * 1000, y_grid[0] * 1000, y_grid[-1] * 1000]
    axes[1].imshow(object_img, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("Предмет: треугольник")
    axes[1].set_xlabel("z, мм")
    axes[1].set_ylabel("y, мм")

    axes[2].imshow(image_img, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("Изображение треугольника")
    axes[2].set_xlabel("z, мм")
    axes[2].set_ylabel("y, мм")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
