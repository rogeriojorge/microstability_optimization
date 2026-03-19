#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "PyVista is required to view VTK files. Install it with 'pip install pyvista'."
    ) from exc


SUPPORTED_EXTENSIONS = {".vtu", ".vts"}
DEFAULT_FILENAMES = ("curves_init.vtu", "surf_init_big.vts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open VTU/VTS coil or surface files in an interactive PyVista window.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files or case directories to display. If the first argument is a case directory and the second is an integer, that integer is used as the maxmode. Maxmode 0 maps to the init pair.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show mesh edges for surface-like datasets.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Render surfaces as wireframes instead of shaded surfaces.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=1.0,
        help="Opacity for plotted meshes, between 0 and 1.",
    )
    return parser.parse_args()


def resolve_default_files() -> list[Path]:
    search_dirs = []
    for candidate in (Path.cwd(), Path(sys.argv[0]).expanduser().resolve().parent):
        if candidate not in search_dirs:
            search_dirs.append(candidate)

    for directory in search_dirs:
        default_paths = [directory / filename for filename in DEFAULT_FILENAMES]
        if all(path.exists() for path in default_paths):
            return default_paths

    searched = ", ".join(str(directory) for directory in search_dirs)
    missing = ", ".join(DEFAULT_FILENAMES)
    raise FileNotFoundError(
        f"Could not find default files {missing} in any of: {searched}"
    )


def candidate_surface_filenames(maxmode: int) -> list[str]:
    if maxmode == 0:
        return ["surf_init_big.vts", "surf_init.vts"]
    return [f"surf_big_opt_maxmode{maxmode}.vts", f"surf_opt_maxmode{maxmode}.vts"]


def default_filenames_for_maxmode(maxmode: int) -> tuple[str, list[str]]:
    if maxmode == 0:
        return ("curves_init.vtu", candidate_surface_filenames(maxmode))
    return (f"curves_opt_maxmode{maxmode}.vtu", candidate_surface_filenames(maxmode))


def resolve_directory_input(raw_path: str, maxmode: int = 0) -> list[Path]:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File or directory not found: {path}")
    if not path.is_dir():
        if maxmode != 0:
            raise ValueError("Maxmode can only be used when the first argument is a directory")
        return [path]

    candidate_dirs = []
    if path.name == "coils":
        candidate_dirs.append(path)

    coils_dir = path / "coils"
    if coils_dir.exists() and coils_dir.is_dir():
        candidate_dirs.append(coils_dir)

    if path not in candidate_dirs:
        candidate_dirs.append(path)

    curve_filename, surface_filenames = default_filenames_for_maxmode(maxmode)
    for directory in candidate_dirs:
        curve_path = directory / curve_filename
        for surface_filename in surface_filenames:
            surface_path = directory / surface_filename
            if curve_path.exists() and surface_path.exists():
                return [curve_path, surface_path]

    searched = ", ".join(str(directory) for directory in candidate_dirs)
    missing = ", ".join([curve_filename] + surface_filenames)
    raise FileNotFoundError(
        f"Could not find default files {missing} in directory input: {path}. Checked: {searched}"
    )


def resolve_input_paths(raw_inputs: list[str]) -> list[str]:
    if not raw_inputs:
        return [str(path) for path in resolve_default_files()]

    if len(raw_inputs) == 2:
        try:
            maxmode = int(raw_inputs[1])
        except ValueError:
            maxmode = None
        if maxmode is not None:
            return [str(path) for path in resolve_directory_input(raw_inputs[0], maxmode=maxmode)]

    resolved_inputs = []
    for raw_input in raw_inputs:
        resolved_inputs.extend(str(path) for path in resolve_directory_input(raw_input))
    return resolved_inputs


def validate_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Expected one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def matching_regular_surface(path: Path) -> Path | None:
    if path.suffix.lower() != ".vts" or "surf_big" not in path.name:
        return None

    regular_name = path.name.replace("surf_big_", "surf_")
    regular_path = path.with_name(regular_name)
    if regular_path.exists():
        return regular_path
    return None


def copy_periodic_scalar_to_big_surface(big_dataset: pv.DataSet, regular_dataset: pv.DataSet, scalar_name: str) -> bool:
    if not hasattr(big_dataset, "dimensions") or not hasattr(regular_dataset, "dimensions"):
        return False

    big_dims = tuple(int(value) for value in big_dataset.dimensions)
    regular_dims = tuple(int(value) for value in regular_dataset.dimensions)
    if len(big_dims) != 3 or len(regular_dims) != 3:
        return False

    regular_phi = regular_dims[1]
    regular_theta = regular_dims[2]
    big_phi = big_dims[1]
    big_theta = big_dims[2]

    if regular_phi <= 0 or regular_theta <= 0:
        return False
    if big_theta != regular_theta + 1:
        return False
    if (big_phi - 1) % regular_phi != 0:
        return False

    phi_periods = (big_phi - 1) // regular_phi
    if phi_periods <= 0:
        return False

    regular_scalar = regular_dataset[scalar_name]
    if len(regular_scalar.shape) != 1:
        return False

    regular_grid = regular_scalar.reshape((regular_theta, regular_phi), order="C")
    regular_grid_closed_theta = np.concatenate([regular_grid, regular_grid[:1, :]], axis=0)
    repeated_grid = np.tile(regular_grid_closed_theta, (1, phi_periods))
    big_grid = np.concatenate([repeated_grid, regular_grid_closed_theta[:, :1]], axis=1)

    big_dataset[scalar_name] = big_grid.reshape(-1, order="C")
    return True


def prepare_dataset(path: Path) -> pv.DataSet:
    dataset = pv.read(path)
    regular_path = matching_regular_surface(path)
    if regular_path is None:
        return dataset

    regular_dataset = pv.read(regular_path)
    if "B.n/B" in dataset.array_names and "B.n/B" in regular_dataset.array_names:
        if copy_periodic_scalar_to_big_surface(dataset, regular_dataset, "B.n/B"):
            print(f"{path.name}: using B.n/B from {regular_path.name} on big geometry")
    return dataset


def choose_scalars(dataset: pv.DataSet) -> str | None:
    ignored_scalar_names = {"Normals", "vtkOriginalPointIds", "idx", "ids", "object_id"}
    scalar_names = [name for name in dataset.array_names if name not in ignored_scalar_names]
    return scalar_names[0] if scalar_names else None


def scalar_limits(dataset: pv.DataSet, scalar_name: str) -> tuple[float, float] | None:
    array = dataset[scalar_name]
    if len(array.shape) != 1:
        return None

    max_abs = float(abs(array).max())
    if max_abs == 0.0:
        return (0.0, 0.0)
    return (-max_abs, max_abs)


def add_dataset(
    plotter: pv.Plotter,
    dataset: pv.DataSet,
    label: str,
    suffix: str,
    show_edges: bool,
    wireframe: bool,
    opacity: float,
) -> None:
    is_line_like = suffix.lower() == ".vtu"
    style = "wireframe" if wireframe and not is_line_like else "surface"
    scalars = choose_scalars(dataset)
    kwargs = {
        "name": label,
        "opacity": opacity,
        "show_scalar_bar": scalars is not None,
    }

    if scalars is not None:
        kwargs["scalars"] = scalars
        kwargs["scalar_bar_args"] = {"title": scalars}
        limits = scalar_limits(dataset, scalars)
        if limits is not None:
            kwargs["clim"] = limits
            kwargs["cmap"] = "coolwarm"
        scalar_array = dataset[scalars]
        print(
            f"{label}: {scalars} min={float(scalar_array.min()):.6g}, "
            f"max={float(scalar_array.max()):.6g}, absmax={float(abs(scalar_array).max()):.6g}"
        )

    if is_line_like:
        kwargs.update(
            {
                "line_width": 4,
                "render_lines_as_tubes": True,
            }
        )
        if scalars is None:
            kwargs["color"] = "tomato"
    else:
        kwargs.update(
            {
                "style": style,
                "show_edges": show_edges,
                "smooth_shading": True,
            }
        )
        if scalars is None:
            kwargs["color"] = "lightsteelblue"

    plotter.add_mesh(dataset, **kwargs)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be between 0 and 1")

    input_paths = resolve_input_paths(args.files)

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid(color="lightgray")

    for raw_path in input_paths:
        path = validate_path(raw_path)
        dataset = prepare_dataset(path)
        add_dataset(
            plotter=plotter,
            dataset=dataset,
            label=path.name,
            suffix=path.suffix,
            show_edges=args.show_edges,
            wireframe=args.wireframe,
            opacity=args.opacity,
        )

    plotter.camera_position = "iso"
    plotter.add_title("VTK coil/surface viewer", font_size=12)
    plotter.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc