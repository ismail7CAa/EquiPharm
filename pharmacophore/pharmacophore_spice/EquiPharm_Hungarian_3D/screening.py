"""SPICE-backed EquiPharm_Hungarian_3D screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_3d_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian_3D", "hungarian_3d", "geometry_distance", **kwargs)

