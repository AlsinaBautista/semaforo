# =============================================================================
# Semáforo Inteligente — Network Graph Manager
# =============================================================================
"""
Manages the traffic network topology as a directed graph.

Provides:
  - Loading topology from JSON configuration.
  - Neighbor lookup for intersections.
  - Shortest path computation between intersections.
  - Corridor identification for green wave coordination.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger("coordinator.graph")


class NetworkGraphManager:
    """
    Manages the traffic network topology using NetworkX.

    The topology is loaded from a JSON file containing intersection nodes
    and road edges with physical properties (distance, speed, lanes).
    """

    def __init__(self, topology_file: str | Path) -> None:
        self._graph = nx.DiGraph()
        self._intersections: dict[str, dict[str, Any]] = {}
        self._roads: dict[str, dict[str, Any]] = {}

        self._load_topology(Path(topology_file))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def intersection_count(self) -> int:
        """Number of intersections in the network."""
        return len(self._intersections)

    @property
    def road_count(self) -> int:
        """Number of road segments in the network."""
        return len(self._roads)

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph."""
        return self._graph

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load_topology(self, topology_file: Path) -> None:
        """
        Load network topology from a JSON configuration file.

        Expected format:
        {
            "network_id": "...",
            "intersections": [ { "id": "...", "name": "...", "location": {...} } ],
            "roads": [ { "id": "...", "source_id": "...", "target_id": "...", ... } ]
        }
        """
        if not topology_file.exists():
            logger.warning(
                "Topology file not found: %s. Starting with empty graph.",
                topology_file,
            )
            return

        with open(topology_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load intersections as nodes
        for intersection in data.get("intersections", []):
            int_id = intersection["id"]
            self._intersections[int_id] = intersection
            self._graph.add_node(
                int_id,
                name=intersection.get("name", ""),
                lat=intersection.get("location", {}).get("latitude", 0.0),
                lng=intersection.get("location", {}).get("longitude", 0.0),
                zone_id=intersection.get("zone_id", "default"),
                num_approaches=intersection.get("num_approaches", 4),
            )

        # Load roads as edges
        for road in data.get("roads", []):
            road_id = road["id"]
            self._roads[road_id] = road
            source = road["source_id"]
            target = road["target_id"]

            self._graph.add_edge(
                source,
                target,
                id=road_id,
                distance_m=road.get("distance_m", 0.0),
                speed_limit_kmh=road.get("speed_limit_kmh", 50.0),
                lanes=road.get("lanes", 2),
                road_class=road.get("road_class", "secondary"),
            )

            # Add reverse edge if the road is bidirectional
            if not road.get("one_way", False):
                self._graph.add_edge(
                    target,
                    source,
                    id=f"{road_id}-rev",
                    distance_m=road.get("distance_m", 0.0),
                    speed_limit_kmh=road.get("speed_limit_kmh", 50.0),
                    lanes=road.get("lanes", 2),
                    road_class=road.get("road_class", "secondary"),
                )

        logger.info(
            "Topology loaded: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_neighbors(self, intersection_id: str) -> list[str]:
        """Get the IDs of all neighboring intersections."""
        if intersection_id not in self._graph:
            return []
        return list(self._graph.neighbors(intersection_id))

    def get_intersection(self, intersection_id: str) -> dict[str, Any] | None:
        """Get the full intersection data by ID."""
        return self._intersections.get(intersection_id)

    def get_edge_distance(self, source_id: str, target_id: str) -> float:
        """Get the distance in meters between two adjacent intersections."""
        try:
            return float(self._graph[source_id][target_id].get("distance_m", 0.0))
        except KeyError:
            logger.warning(
                "No edge found between %s and %s", source_id, target_id
            )
            return 0.0

    def get_edge_speed_limit(self, source_id: str, target_id: str) -> float:
        """Get the speed limit (km/h) for the road between two intersections."""
        try:
            return float(
                self._graph[source_id][target_id].get("speed_limit_kmh", 50.0)
            )
        except KeyError:
            return 50.0

    # -------------------------------------------------------------------------
    # Path Finding
    # -------------------------------------------------------------------------

    def shortest_path(
        self, source_id: str, target_id: str, weight: str = "distance_m"
    ) -> list[str]:
        """
        Compute the shortest path between two intersections.

        Args:
            source_id: Starting intersection ID.
            target_id: Destination intersection ID.
            weight: Edge attribute to use as weight (default: distance_m).

        Returns:
            Ordered list of intersection IDs along the shortest path.
            Empty list if no path exists.
        """
        try:
            return list(
                nx.shortest_path(self._graph, source_id, target_id, weight=weight)
            )
        except nx.NetworkXNoPath:
            logger.warning(
                "No path found from %s to %s", source_id, target_id
            )
            return []
        except nx.NodeNotFound as e:
            logger.warning("Node not found: %s", e)
            return []

    def shortest_path_length(
        self, source_id: str, target_id: str, weight: str = "distance_m"
    ) -> float:
        """Compute the shortest path distance between two intersections."""
        try:
            return float(
                nx.shortest_path_length(
                    self._graph, source_id, target_id, weight=weight
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    # -------------------------------------------------------------------------
    # Corridor Identification
    # -------------------------------------------------------------------------

    def identify_corridors(self) -> list[list[str]]:
        """
        Identify linear corridors in the network for green wave coordination.

        A corridor is a sequence of intersections connected by primary or
        secondary roads that form a roughly linear path. This uses a
        simplified heuristic: find chains of nodes with degree <= 2 along
        primary roads, or the longest simple paths in the network.

        Returns:
            List of corridors, each being an ordered list of intersection IDs.
        """
        corridors: list[list[str]] = []

        # Simple heuristic: find all paths through primary roads
        primary_edges = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("road_class") == "primary"
        ]

        if not primary_edges:
            # Fallback: use all edges
            primary_edges = list(self._graph.edges())

        # Build subgraph of primary roads
        primary_graph = self._graph.edge_subgraph(primary_edges).copy()

        # Find connected components and create corridors from each
        for component in nx.weakly_connected_components(primary_graph):
            subgraph = primary_graph.subgraph(component)
            # Find endpoints (nodes with in-degree or out-degree of 0/1)
            endpoints = [
                n
                for n in subgraph.nodes()
                if subgraph.in_degree(n) == 0 or subgraph.out_degree(n) == 0
            ]

            if len(endpoints) >= 2:
                # Find path between first and last endpoint
                try:
                    path = nx.shortest_path(
                        subgraph, endpoints[0], endpoints[-1], weight="distance_m"
                    )
                    corridors.append(path)
                except nx.NetworkXNoPath:
                    # If no path, just add nodes in order
                    corridors.append(sorted(component))
            elif component:
                corridors.append(sorted(component))

        logger.info("Identified %d corridors", len(corridors))
        return corridors

    def get_all_intersection_ids(self) -> list[str]:
        """Get all intersection IDs in the network."""
        return list(self._intersections.keys())
