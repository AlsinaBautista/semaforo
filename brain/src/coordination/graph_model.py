# Semáforo Inteligente - Brain Module
"""Directed graph model of the traffic network.

Provides a lightweight graph structure (no external dependency on
networkx) for representing intersections and the roads connecting them.
Supports JSON serialisation for interoperability with the firmware and
dashboard layers.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class IntersectionNode:
    """A node representing a traffic-signal intersection.

    Attributes:
        id: Unique identifier string (e.g. ``"tl_0"``).
        lat: Latitude (decimal degrees), optional.
        lon: Longitude (decimal degrees), optional.
        metadata: Arbitrary extra data.
    """

    id: str
    lat: float = 0.0
    lon: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadEdge:
    """A directed edge representing a road segment between intersections.

    Attributes:
        source: Source intersection id.
        target: Target intersection id.
        length_m: Road length in metres.
        speed_limit_kmh: Speed limit in km/h.
        lanes: Number of lanes.
        metadata: Arbitrary extra data.
    """

    source: str
    target: str
    length_m: float = 0.0
    speed_limit_kmh: float = 50.0
    lanes: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def travel_time(self) -> float:
        """Estimated free-flow travel time in seconds.

        Returns:
            Travel time computed from length and speed limit.
        """
        if self.speed_limit_kmh <= 0:
            return float("inf")
        return self.length_m / (self.speed_limit_kmh / 3.6)


class TrafficNetwork:
    """Directed graph representation of a traffic-signal network.

    Nodes are :class:`IntersectionNode` instances and edges are
    :class:`RoadEdge` instances.

    Example::

        net = TrafficNetwork()
        net.add_intersection("A", lat=-34.6, lon=-58.4)
        net.add_intersection("B", lat=-34.6, lon=-58.39)
        net.add_road("A", "B", length_m=500, speed_limit_kmh=40)
        print(net.neighbors("A"))  # ['B']
    """

    def __init__(self) -> None:
        self._nodes: dict[str, IntersectionNode] = {}
        self._edges: dict[str, list[RoadEdge]] = {}  # adjacency list

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_intersection(
        self,
        node_id: str,
        lat: float = 0.0,
        lon: float = 0.0,
        **metadata: Any,
    ) -> IntersectionNode:
        """Add an intersection node to the network.

        Args:
            node_id: Unique node identifier.
            lat: Latitude.
            lon: Longitude.
            **metadata: Additional attributes stored in the node.

        Returns:
            The created :class:`IntersectionNode`.
        """
        node = IntersectionNode(id=node_id, lat=lat, lon=lon, metadata=metadata)
        self._nodes[node_id] = node
        self._edges.setdefault(node_id, [])
        return node

    def get_intersection(self, node_id: str) -> IntersectionNode:
        """Retrieve a node by id.

        Args:
            node_id: Node identifier.

        Returns:
            The matching :class:`IntersectionNode`.

        Raises:
            KeyError: If the node does not exist.
        """
        return self._nodes[node_id]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_road(
        self,
        source: str,
        target: str,
        length_m: float = 0.0,
        speed_limit_kmh: float = 50.0,
        lanes: int = 1,
        **metadata: Any,
    ) -> RoadEdge:
        """Add a directed road edge between two intersections.

        Args:
            source: Source node id.
            target: Target node id.
            length_m: Road length in metres.
            speed_limit_kmh: Speed limit (km/h).
            lanes: Number of lanes.
            **metadata: Additional edge attributes.

        Returns:
            The created :class:`RoadEdge`.

        Raises:
            KeyError: If source or target nodes do not exist.
        """
        if source not in self._nodes:
            raise KeyError(f"Source node '{source}' not found.")
        if target not in self._nodes:
            raise KeyError(f"Target node '{target}' not found.")

        edge = RoadEdge(
            source=source,
            target=target,
            length_m=length_m,
            speed_limit_kmh=speed_limit_kmh,
            lanes=lanes,
            metadata=metadata,
        )
        self._edges[source].append(edge)
        return edge

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[str]:
        """Return ids of nodes directly reachable from *node_id*.

        Args:
            node_id: Source node identifier.

        Returns:
            List of neighbor node ids.
        """
        return [e.target for e in self._edges.get(node_id, [])]

    def shortest_path(
        self, source: str, target: str
    ) -> tuple[list[str], float]:
        """Compute the shortest path by travel time (Dijkstra).

        Args:
            source: Start node id.
            target: End node id.

        Returns:
            A tuple ``(path, total_travel_time)`` where *path* is the
            ordered list of node ids and *total_travel_time* is in
            seconds.

        Raises:
            KeyError: If source or target do not exist.
            ValueError: If no path exists.
        """
        if source not in self._nodes:
            raise KeyError(f"Source '{source}' not in network.")
        if target not in self._nodes:
            raise KeyError(f"Target '{target}' not in network.")

        dist: dict[str, float] = {n: float("inf") for n in self._nodes}
        prev: dict[str, str | None] = {n: None for n in self._nodes}
        dist[source] = 0.0
        pq: list[tuple[float, str]] = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for edge in self._edges.get(u, []):
                alt = d + edge.travel_time
                if alt < dist[edge.target]:
                    dist[edge.target] = alt
                    prev[edge.target] = u
                    heapq.heappush(pq, (alt, edge.target))

        if dist[target] == float("inf"):
            raise ValueError(f"No path from '{source}' to '{target}'.")

        # Reconstruct path.
        path: list[str] = []
        node: str | None = target
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, dist[target]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the network to a plain dict.

        Returns:
            Dictionary with ``nodes`` and ``edges`` lists.
        """
        return {
            "nodes": [asdict(n) for n in self._nodes.values()],
            "edges": [
                asdict(e) for edges in self._edges.values() for e in edges
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save the network to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> TrafficNetwork:
        """Load a network from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Reconstructed :class:`TrafficNetwork`.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        net = cls()
        for node_data in data.get("nodes", []):
            meta = node_data.pop("metadata", {})
            net.add_intersection(**node_data, **meta)

        for edge_data in data.get("edges", []):
            meta = edge_data.pop("metadata", {})
            net.add_road(**edge_data, **meta)

        return net

    def __repr__(self) -> str:
        n_edges = sum(len(el) for el in self._edges.values())
        return (
            f"TrafficNetwork(nodes={len(self._nodes)}, edges={n_edges})"
        )
