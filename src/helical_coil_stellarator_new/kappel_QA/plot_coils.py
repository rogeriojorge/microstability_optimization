#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def default_filenames_for_maxmode(maxmode: int) -> tuple[str, str]:
    if maxmode == 0:
        return DEFAULT_FILENAMES
    return (f"curves_opt_maxmode{maxmode}.vtu", f"surf_big_opt_maxmode{maxmode}.vts")


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

    filenames = default_filenames_for_maxmode(maxmode)
    for directory in candidate_dirs:
        default_paths = [directory / filename for filename in filenames]
        if all(candidate.exists() for candidate in default_paths):
            return default_paths

    searched = ", ".join(str(directory) for directory in candidate_dirs)
    missing = ", ".join(filenames)
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


def choose_scalars(dataset: pv.DataSet) -> str | None:
    scalar_names = [name for name in dataset.array_names if name not in {"Normals", "vtkOriginalPointIds"}]
    return scalar_names[0] if scalar_names else None


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
        dataset = pv.read(path)
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