import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Slider
import json
from pathlib import Path as FsPath


def thin_lens_geometry(f, lens_x, x_object):
    s = lens_x - x_object
    denom = (1.0 / f - 1.0 / s)
    if abs(denom) < 1e-12:
        raise ValueError("Положение объекта совпало с фокальной плоскостью линзы.")
    s_img = 1.0 / denom
    m = -s_img / s
    return s, s_img, m


def system_geometry(x_object, lens1, lens2=None):
    _, s1_img, m1 = thin_lens_geometry(lens1["focal_length"], lens1["x_position"], x_object)
    x_image1 = lens1["x_position"] + s1_img
    if lens2 is None or not lens2.get("enabled", False):
        details = {
            "m1": m1,
            "m2": 1.0,
        }
        return x_image1, m1, details

    _, s2_img, m2 = thin_lens_geometry(lens2["focal_length"], lens2["x_position"], x_image1)
    x_image2 = lens2["x_position"] + s2_img
    details = {
        "m1": m1,
        "m2": m2,
    }
    return x_image2, m1 * m2, details


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


def trace_ray_path(y0, theta0, x_object, lenses, x_image):
    x_points = [x_object]
    y_points = [y0]
    x_curr = x_object
    y_curr = y0
    theta = theta0
    for lens in lenses:
        dx = lens["x_position"] - x_curr
        y_curr = y_curr + theta * dx
        x_curr = lens["x_position"]
        x_points.append(x_curr)
        y_points.append(y_curr)
        theta = theta - y_curr / lens["focal_length"]

    y_curr = y_curr + theta * (x_image - x_curr)
    x_points.append(x_image)
    y_points.append(y_curr)
    return x_points, y_points


def trace_sample_rays(ax, x_object, y_sources, lenses, x_image, object_height, magnification):
    first_lens_x = lenses[0]["x_position"]
    for y0 in y_sources:
        x1, y1 = trace_ray_path(y0, 0.0, x_object, lenses, x_image)
        ax.plot(x1, y1, "tab:blue", lw=1.2)

        theta_center = -y0 / (first_lens_x - x_object)
        x2, y2 = trace_ray_path(y0, theta_center, x_object, lenses, x_image)
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
    for i, lens in enumerate(lenses, start=1):
        label = f"Линза {i}"
        ax.axvline(lens["x_position"], color="k", ls="--", lw=1.2, label=label)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_title("Ход лучей (тонкие линзы)")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)


def draw_scene(
    axes,
    lens1_x,
    x_object,
    lens1,
    lens2,
    y_sources,
    obj_vertices,
    y_grid,
    z_grid,
):
    lens1["x_position"] = float(lens1_x)
    x_image, magnification, details = system_geometry(x_object, lens1, lens2)
    lenses = [lens1]
    if lens2.get("enabled", False):
        lenses.append(lens2)

    axes[0].clear()
    trace_sample_rays(
        axes[0],
        x_object=x_object,
        y_sources=y_sources,
        lenses=lenses,
        x_image=x_image,
        object_height=np.max(obj_vertices[:, 0]) - np.min(obj_vertices[:, 0]),
        magnification=magnification,
    )

    image_vertices = magnification * obj_vertices
    object_img = rasterize_triangle(obj_vertices, y_grid, z_grid)
    image_img = rasterize_triangle(image_vertices, y_grid, z_grid)

    extent = [z_grid[0] * 1000, z_grid[-1] * 1000, y_grid[0] * 1000, y_grid[-1] * 1000]

    axes[1].clear()
    axes[1].imshow(object_img, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    object_height = np.max(obj_vertices[:, 0]) - np.min(obj_vertices[:, 0])
    axes[1].set_title(f"Предмет: треугольник, \n {object_height=}")
    axes[1].set_xlabel("z, мм")
    axes[1].set_ylabel("y, мм")

    axes[2].clear()
    axes[2].imshow(image_img, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    new_height = object_height * magnification
    axes[2].set_title(f"Изображение треугольника, \n {new_height=}")
    axes[2].set_xlabel("z, мм")
    axes[2].set_ylabel("y, мм")

    title = f"x1={lens1['x_position']:.4f} м, x_изобр={x_image:.4f} м, m={magnification:.3f}"
    return title


def load_config(path="config.json"):
    defaults = {
        "lens": {
            "focal_length": 0.060,
            "x_position": 0.0,
        },
        "second_lens": {
            "enabled": True,
            "focal_length": 0.120,
            "x_position": 0.160,
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
            "lens1_slider_min": -0.08,
            "lens1_slider_max": 0.08,
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
    x_object = cfg["object"]["x_position"]
    lens1 = cfg["lens"]
    lens2 = cfg["second_lens"]
    x_image, m, details = system_geometry(x_object, lens1, lens2)

    print("=== Система тонких линз ===")
    print(
        f"Линза 1: f1 = {lens1['focal_length']*1000:.1f} мм, x1 = {lens1['x_position']:.4f} м"
    )
    if lens2.get("enabled", False):
        print(
            f"Линза 2: f2 = {lens2['focal_length']*1000:.1f} мм, x2 = {lens2['x_position']:.4f} м"
        )
        print(f"Увеличения по каскаду: m1 = {details['m1']:.3f}, m2 = {details['m2']:.3f}")
    print(f"Общее увеличение: m = {m:.3f}")
    print(f"Объект на графике лучей: x = {x_object:.4f} м")
    print(f"Проекция объекта на графике лучей: x = {x_image:.4f} м")

    obj_vertices = build_triangle(
        height=cfg["object"]["triangle_height"],
        width=cfg["object"]["triangle_width"],
    )

    y_range = cfg["plot"]["y_range"]
    z_range = cfg["plot"]["z_range"]
    n_grid = int(cfg["plot"]["grid_points"])
    y_grid = np.linspace(-y_range, y_range, n_grid)
    z_grid = np.linspace(-z_range, z_range, n_grid)
    y_sources = np.linspace(
        cfg["plot"]["ray_y_min"],
        cfg["plot"]["ray_y_max"],
        int(cfg["plot"]["ray_count"]),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(cfg["plot"]["fig_width_in"], cfg["plot"]["fig_height_in"]),
    )
    fig.subplots_adjust(bottom=0.20)

    title = draw_scene(
        axes=axes,
        lens1_x=lens1["x_position"],
        x_object=x_object,
        lens1=lens1,
        lens2=lens2,
        y_sources=y_sources,
        obj_vertices=obj_vertices,
        y_grid=y_grid,
        z_grid=z_grid,
    )
    suptitle = fig.suptitle(title)

    slider_ax = fig.add_axes([0.18, 0.06, 0.64, 0.05])
    slider = Slider(
        ax=slider_ax,
        label="x первой линзы, м",
        valmin=float(cfg["plot"]["lens1_slider_min"]),
        valmax=float(cfg["plot"]["lens1_slider_max"]),
        valinit=float(lens1["x_position"]),
    )

    def on_slider_change(val):
        try:
            title_new = draw_scene(
                axes=axes,
                lens1_x=float(val),
                x_object=x_object,
                lens1=lens1,
                lens2=lens2,
                y_sources=y_sources,
                obj_vertices=obj_vertices,
                y_grid=y_grid,
                z_grid=z_grid,
            )
            suptitle.set_text(title_new)
        except ValueError:
            axes[0].clear()
            axes[0].set_title("Невозможная конфигурация (фокальная плоскость)")
            axes[0].grid(alpha=0.25)
            suptitle.set_text("Попробуйте другое положение первой линзы")
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
