"""
network_generator.py — Programmatic SUMO network generation.

Semáforo Inteligente Project

This module provides a ``NetworkGenerator`` class that can:
  1. Generate a single 4-way intersection from scratch (.nod.xml, .edg.xml,
     then compile via ``netconvert``).
  2. Generate a corridor of N evenly-spaced intersections (placeholder).
  3. Import a real-world network from an OpenStreetMap (.osm) file.
  4. Generate a SUMO routes file from a demand specification dict.

All heavy lifting is done by calling SUMO's ``netconvert`` as a subprocess,
so SUMO must be installed and on the PATH (or pointed to via ``SUMO_HOME``).

Example
-------
>>> from network_generator import NetworkGenerator
>>> gen = NetworkGenerator()
>>> gen.generate_single_intersection("output/single")
>>> gen.generate_routes(
...     net_file="output/single/intersection.net.xml",
...     output_dir="output/single",
...     demands={
...         "ns_through": 600, "sn_through": 500,
...         "ew_through": 400, "we_through": 350,
...     },
... )
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML Templates
# ---------------------------------------------------------------------------

_NODES_HEADER = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    """)

_EDGES_HEADER = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    """)

_ROUTES_HEADER = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    """)


class NetworkGenerator:
    """Generates and manages SUMO network files programmatically.

    Parameters
    ----------
    sumo_binary : str, optional
        Path to the ``netconvert`` binary.  Defaults to ``"netconvert"``
        (i.e., it must be on the PATH or located via ``SUMO_HOME``).
    """

    def __init__(self, sumo_binary: str = "netconvert") -> None:
        self.sumo_binary = sumo_binary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dir(path: str | Path) -> Path:
        """Create *path* (and parents) if it doesn't exist; return as Path."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _run_netconvert(self, args: List[str], cwd: Optional[str] = None) -> None:
        """Run ``netconvert`` with the given CLI arguments.

        Raises
        ------
        FileNotFoundError
            If ``netconvert`` is not installed / not on PATH.
        subprocess.CalledProcessError
            If ``netconvert`` exits with a non-zero return code.
        """
        cmd = [self.sumo_binary] + args
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.debug("netconvert stdout:\n%s", result.stdout)
        if result.returncode != 0:
            logger.error("netconvert stderr:\n%s", result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        logger.info("netconvert completed successfully.")

    # ------------------------------------------------------------------
    # 1. Single intersection
    # ------------------------------------------------------------------

    def generate_single_intersection(
        self,
        output_dir: str,
        *,
        num_lanes: int = 2,
        approach_length: float = 200.0,
        speed_kmh: float = 50.0,
    ) -> str:
        """Generate a 4-way, traffic-light-controlled intersection.

        Creates ``intersection.nod.xml``, ``intersection.edg.xml`` in
        *output_dir* and compiles them into ``intersection.net.xml`` via
        ``netconvert``.

        Parameters
        ----------
        output_dir : str
            Directory where the generated files will be placed.
        num_lanes : int
            Number of lanes per direction on each approach.
        approach_length : float
            Distance (metres) from the centre to each approach endpoint.
        speed_kmh : float
            Speed limit in km/h (converted to m/s for SUMO).

        Returns
        -------
        str
            Absolute path to the generated ``.net.xml`` file.
        """
        out = self._ensure_dir(output_dir)
        speed_ms = round(speed_kmh / 3.6, 2)

        # -- Nodes ----------------------------------------------------------
        directions: Dict[str, Tuple[float, float]] = {
            "north": (0.0, approach_length),
            "south": (0.0, -approach_length),
            "east": (approach_length, 0.0),
            "west": (-approach_length, 0.0),
        }
        nod_path = out / "intersection.nod.xml"
        with open(nod_path, "w", encoding="utf-8") as f:
            f.write(_NODES_HEADER)
            f.write(f'    <node id="center" x="0.0" y="0.0" type="traffic_light"/>\n')
            for name, (x, y) in directions.items():
                f.write(f'    <node id="{name}" x="{x}" y="{y}" type="priority"/>\n')
            f.write("</nodes>\n")
        logger.info("Wrote %s", nod_path)

        # -- Edges ----------------------------------------------------------
        edg_path = out / "intersection.edg.xml"
        with open(edg_path, "w", encoding="utf-8") as f:
            f.write(_EDGES_HEADER)
            for name in directions:
                f.write(
                    f'    <edge id="{name}_in"  from="{name}" to="center" '
                    f'numLanes="{num_lanes}" speed="{speed_ms}"/>\n'
                )
                f.write(
                    f'    <edge id="{name}_out" from="center" to="{name}" '
                    f'numLanes="{num_lanes}" speed="{speed_ms}"/>\n'
                )
            f.write("</edges>\n")
        logger.info("Wrote %s", edg_path)

        # -- Compile network ------------------------------------------------
        net_path = out / "intersection.net.xml"
        self._run_netconvert(
            [
                f"--node-files={nod_path}",
                f"--edge-files={edg_path}",
                f"--output-file={net_path}",
                "--tls.guess", "true",
            ]
        )
        logger.info("Network compiled → %s", net_path)
        return str(net_path)

    # ------------------------------------------------------------------
    # 2. Corridor of N intersections (placeholder — Phase 4)
    # ------------------------------------------------------------------

    def generate_corridor(
        self,
        output_dir: str,
        num_intersections: int = 3,
        spacing_m: float = 300.0,
        *,
        num_lanes: int = 2,
        speed_kmh: float = 50.0,
        approach_length: float = 200.0,
    ) -> str:
        """Generate a linear corridor of *num_intersections* signalised junctions.

        .. note::
            **Phase 4 placeholder** — this method creates the node/edge files
            for a straight East-West corridor with North-South side streets at
            each intersection, then compiles via ``netconvert``.

        Parameters
        ----------
        output_dir : str
            Target directory for generated files.
        num_intersections : int
            Number of intersections along the corridor.
        spacing_m : float
            Distance (m) between consecutive intersections.
        num_lanes : int
            Lanes per direction on the main corridor and side streets.
        speed_kmh : float
            Speed limit (km/h) on the corridor.
        approach_length : float
            Length (m) of each side-street approach.

        Returns
        -------
        str
            Absolute path to the generated ``.net.xml``.
        """
        out = self._ensure_dir(output_dir)
        speed_ms = round(speed_kmh / 3.6, 2)

        # -- Nodes ----------------------------------------------------------
        nod_path = out / "corridor.nod.xml"
        with open(nod_path, "w", encoding="utf-8") as f:
            f.write(_NODES_HEADER)

            # West entry
            f.write(f'    <node id="west_entry" x="{-approach_length}" y="0.0" type="priority"/>\n')

            for i in range(num_intersections):
                cx = i * spacing_m
                f.write(f'    <node id="c{i}" x="{cx}" y="0.0" type="traffic_light"/>\n')
                f.write(f'    <node id="c{i}_north" x="{cx}" y="{approach_length}" type="priority"/>\n')
                f.write(f'    <node id="c{i}_south" x="{cx}" y="{-approach_length}" type="priority"/>\n')

            # East entry
            last_x = (num_intersections - 1) * spacing_m + approach_length
            f.write(f'    <node id="east_entry" x="{last_x}" y="0.0" type="priority"/>\n')

            f.write("</nodes>\n")

        # -- Edges ----------------------------------------------------------
        edg_path = out / "corridor.edg.xml"
        with open(edg_path, "w", encoding="utf-8") as f:
            f.write(_EDGES_HEADER)

            # West entry ↔ first intersection
            f.write(f'    <edge id="west_entry_to_c0" from="west_entry" to="c0" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
            f.write(f'    <edge id="c0_to_west_entry" from="c0" to="west_entry" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')

            for i in range(num_intersections):
                # Side streets
                f.write(f'    <edge id="c{i}_north_in"  from="c{i}_north" to="c{i}" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
                f.write(f'    <edge id="c{i}_north_out" from="c{i}" to="c{i}_north" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
                f.write(f'    <edge id="c{i}_south_in"  from="c{i}_south" to="c{i}" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
                f.write(f'    <edge id="c{i}_south_out" from="c{i}" to="c{i}_south" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')

                # Corridor links between consecutive intersections
                if i < num_intersections - 1:
                    f.write(f'    <edge id="c{i}_to_c{i+1}" from="c{i}" to="c{i+1}" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
                    f.write(f'    <edge id="c{i+1}_to_c{i}" from="c{i+1}" to="c{i}" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')

            # Last intersection ↔ east entry
            last_i = num_intersections - 1
            f.write(f'    <edge id="c{last_i}_to_east_entry" from="c{last_i}" to="east_entry" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')
            f.write(f'    <edge id="east_entry_to_c{last_i}" from="east_entry" to="c{last_i}" numLanes="{num_lanes}" speed="{speed_ms}"/>\n')

            f.write("</edges>\n")

        # -- Compile --------------------------------------------------------
        net_path = out / "corridor.net.xml"
        self._run_netconvert(
            [
                f"--node-files={nod_path}",
                f"--edge-files={edg_path}",
                f"--output-file={net_path}",
                "--tls.guess", "true",
            ]
        )
        logger.info("Corridor network compiled → %s", net_path)
        return str(net_path)

    # ------------------------------------------------------------------
    # 3. Import from OpenStreetMap
    # ------------------------------------------------------------------

    def import_from_osm(
        self,
        osm_file: str,
        output_dir: str,
        *,
        tls_guess: bool = True,
        keep_edges_by_type: Optional[str] = None,
    ) -> str:
        """Import a SUMO network from an OpenStreetMap ``.osm`` file.

        Uses ``netconvert --osm-files`` under the hood.

        Parameters
        ----------
        osm_file : str
            Path to the ``.osm`` or ``.osm.xml`` file.
        output_dir : str
            Directory for the generated ``.net.xml``.
        tls_guess : bool
            Whether to let ``netconvert`` guess traffic-light positions.
        keep_edges_by_type : str, optional
            Comma-separated list of OSM highway types to keep, e.g.
            ``"highway.primary,highway.secondary"``.  If ``None``, all
            types are imported.

        Returns
        -------
        str
            Absolute path to the generated ``.net.xml``.
        """
        out = self._ensure_dir(output_dir)
        osm_path = Path(osm_file).resolve()
        if not osm_path.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_path}")

        net_path = out / "osm_network.net.xml"
        args = [
            f"--osm-files={osm_path}",
            f"--output-file={net_path}",
        ]
        if tls_guess:
            args += ["--tls.guess", "true"]
        if keep_edges_by_type:
            args += [f"--keep-edges.by-type", keep_edges_by_type]

        self._run_netconvert(args)
        logger.info("OSM network compiled → %s", net_path)
        return str(net_path)

    # ------------------------------------------------------------------
    # 4. Route generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_routes(
        net_file: str,
        output_dir: str,
        demands: Dict[str, float],
        *,
        sim_begin: float = 0.0,
        sim_end: float = 3600.0,
        left_turn_ratio: float = 0.15,
        right_turn_ratio: float = 0.10,
    ) -> str:
        """Generate a SUMO routes file from a demand specification.

        Parameters
        ----------
        net_file : str
            Path to the compiled ``.net.xml`` (currently used only for
            documentation; route edge IDs follow the naming convention).
        output_dir : str
            Directory where ``intersection.rou.xml`` will be written.
        demands : dict[str, float]
            Mapping of movement name to vehicles-per-hour for *through*
            movements.  Expected keys follow the pattern
            ``"<origin><destination>_through"`` (e.g., ``"ns_through"``).
            Left- and right-turn flows are derived automatically using
            *left_turn_ratio* and *right_turn_ratio*.
        sim_begin : float
            Simulation start time (seconds).
        sim_end : float
            Simulation end time (seconds).
        left_turn_ratio : float
            Fraction of through demand assigned to left turns.
        right_turn_ratio : float
            Fraction of through demand assigned to right turns.

        Returns
        -------
        str
            Absolute path to the generated ``.rou.xml``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rou_path = out / "intersection.rou.xml"

        # Mapping from through-movement key prefix to (origin_edge, through_out, left_out, right_out)
        # Convention: right-hand traffic
        _TURN_MAP: Dict[str, Tuple[str, str, str, str]] = {
            "ns": ("north_in", "south_out", "east_out", "west_out"),
            "sn": ("south_in", "north_out", "west_out", "east_out"),
            "ew": ("east_in", "west_out", "south_out", "north_out"),
            "we": ("west_in", "east_out", "north_out", "south_out"),
        }

        with open(rou_path, "w", encoding="utf-8") as f:
            f.write(_ROUTES_HEADER)

            # Vehicle type
            f.write(
                '    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" '
                'length="5.0" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>\n\n'
            )

            # Routes & flows
            flow_id = 0
            for key, (origin, through_out, left_out, right_out) in _TURN_MAP.items():
                through_key = f"{key}_through"
                through_vph = demands.get(through_key, 0.0)
                if through_vph <= 0:
                    continue

                left_vph = through_vph * left_turn_ratio
                right_vph = through_vph * right_turn_ratio

                movements = [
                    ("through", through_out, through_vph),
                    ("left", left_out, left_vph),
                    ("right", right_out, right_vph),
                ]

                for movement, out_edge, vph in movements:
                    route_id = f"{key}_{movement}"
                    prob = round(vph / 3600.0, 4)
                    f.write(f'    <route id="{route_id}" edges="{origin} {out_edge}"/>\n')
                    f.write(
                        f'    <flow id="flow_{flow_id}_{route_id}" type="passenger" '
                        f'route="{route_id}" begin="{sim_begin}" end="{sim_end}" '
                        f'probability="{prob}" departLane="best" departSpeed="max"/>\n'
                    )
                    flow_id += 1
                f.write("\n")

            f.write("</routes>\n")

        logger.info("Routes file written → %s", rou_path)
        return str(rou_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Quick CLI for generating the default single-intersection network."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate SUMO network files for Semáforo Inteligente."
    )
    sub = parser.add_subparsers(dest="command")

    # -- single --
    p_single = sub.add_parser("single", help="Generate a single 4-way intersection.")
    p_single.add_argument(
        "-o", "--output-dir",
        default="single_intersection",
        help="Output directory (default: single_intersection).",
    )
    p_single.add_argument("--lanes", type=int, default=2, help="Lanes per direction.")
    p_single.add_argument("--length", type=float, default=200.0, help="Approach length (m).")
    p_single.add_argument("--speed", type=float, default=50.0, help="Speed limit (km/h).")

    # -- corridor --
    p_corridor = sub.add_parser("corridor", help="Generate a corridor of intersections.")
    p_corridor.add_argument(
        "-o", "--output-dir",
        default="corridor",
        help="Output directory (default: corridor).",
    )
    p_corridor.add_argument("-n", "--num", type=int, default=3, help="Number of intersections.")
    p_corridor.add_argument("--spacing", type=float, default=300.0, help="Spacing (m).")

    # -- osm --
    p_osm = sub.add_parser("osm", help="Import network from OpenStreetMap.")
    p_osm.add_argument("osm_file", help="Path to .osm file.")
    p_osm.add_argument(
        "-o", "--output-dir",
        default="osm_network",
        help="Output directory (default: osm_network).",
    )

    args = parser.parse_args()
    gen = NetworkGenerator()

    if args.command == "single":
        net = gen.generate_single_intersection(
            args.output_dir,
            num_lanes=args.lanes,
            approach_length=args.length,
            speed_kmh=args.speed,
        )
        gen.generate_routes(
            net_file=net,
            output_dir=args.output_dir,
            demands={
                "ns_through": 600,
                "sn_through": 500,
                "ew_through": 400,
                "we_through": 350,
            },
        )
        print(f"✅ Single intersection generated in {args.output_dir}/")

    elif args.command == "corridor":
        gen.generate_corridor(
            args.output_dir,
            num_intersections=args.num,
            spacing_m=args.spacing,
        )
        print(f"✅ Corridor generated in {args.output_dir}/")

    elif args.command == "osm":
        gen.import_from_osm(args.osm_file, args.output_dir)
        print(f"✅ OSM network generated in {args.output_dir}/")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
