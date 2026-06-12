# =============================================================================
# Semáforo Inteligente — Tests for NetworkGraphManager
# =============================================================================
"""
Tests for the NetworkGraphManager class.

Covers topology loading, graph queries, pathfinding, corridor identification,
bidirectional edges, missing files, and edge-case handling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.graph_manager import NetworkGraphManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_topology(
    intersections: list[dict] | None = None,
    roads: list[dict] | None = None,
    network_id: str = "test-network",
) -> dict:
    """Helper to build a topology dict."""
    if intersections is None:
        intersections = [
            {
                "id": "int-001",
                "name": "Intersection A",
                "location": {"latitude": -34.600, "longitude": -58.400},
                "num_approaches": 4,
                "has_signal": True,
                "zone_id": "zone-a",
            },
            {
                "id": "int-002",
                "name": "Intersection B",
                "location": {"latitude": -34.601, "longitude": -58.399},
                "num_approaches": 4,
                "has_signal": True,
                "zone_id": "zone-a",
            },
            {
                "id": "int-003",
                "name": "Intersection C",
                "location": {"latitude": -34.602, "longitude": -58.398},
                "num_approaches": 4,
                "has_signal": True,
                "zone_id": "zone-a",
            },
        ]
    if roads is None:
        roads = [
            {
                "id": "road-001",
                "source_id": "int-001",
                "target_id": "int-002",
                "distance_m": 350.0,
                "speed_limit_kmh": 50.0,
                "lanes": 3,
                "one_way": False,
                "road_class": "primary",
            },
            {
                "id": "road-002",
                "source_id": "int-002",
                "target_id": "int-003",
                "distance_m": 350.0,
                "speed_limit_kmh": 50.0,
                "lanes": 3,
                "one_way": False,
                "road_class": "primary",
            },
        ]
    return {"network_id": network_id, "intersections": intersections, "roads": roads}


@pytest.fixture()
def topology_file(tmp_path: Path) -> Path:
    """Create a temporary topology JSON with 3 linear intersections."""
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(_make_topology()), encoding="utf-8")
    return path


@pytest.fixture()
def manager(topology_file: Path) -> NetworkGraphManager:
    """Return a NetworkGraphManager loaded with the 3-node topology."""
    return NetworkGraphManager(topology_file)


# ---------------------------------------------------------------------------
# Tests — Loading & Properties
# ---------------------------------------------------------------------------

class TestLoadingAndProperties:
    """Verify that topology loading populates counts correctly."""

    def test_intersection_count(self, manager: NetworkGraphManager) -> None:
        assert manager.intersection_count == 3

    def test_road_count(self, manager: NetworkGraphManager) -> None:
        assert manager.road_count == 2

    def test_graph_node_count(self, manager: NetworkGraphManager) -> None:
        assert manager.graph.number_of_nodes() == 3

    def test_graph_edge_count_bidirectional(self, manager: NetworkGraphManager) -> None:
        """Two bidirectional roads → 4 directed edges total."""
        assert manager.graph.number_of_edges() == 4


# ---------------------------------------------------------------------------
# Tests — get_all_intersection_ids
# ---------------------------------------------------------------------------

class TestGetAllIntersectionIds:
    def test_returns_all_ids(self, manager: NetworkGraphManager) -> None:
        ids = manager.get_all_intersection_ids()
        assert sorted(ids) == ["int-001", "int-002", "int-003"]

    def test_return_type_is_list(self, manager: NetworkGraphManager) -> None:
        assert isinstance(manager.get_all_intersection_ids(), list)


# ---------------------------------------------------------------------------
# Tests — get_intersection
# ---------------------------------------------------------------------------

class TestGetIntersection:
    def test_existing_intersection(self, manager: NetworkGraphManager) -> None:
        info = manager.get_intersection("int-001")
        assert info is not None
        assert info["id"] == "int-001"
        assert info["name"] == "Intersection A"
        assert info["zone_id"] == "zone-a"

    def test_intersection_location(self, manager: NetworkGraphManager) -> None:
        info = manager.get_intersection("int-002")
        assert info is not None
        assert info["location"]["latitude"] == pytest.approx(-34.601)
        assert info["location"]["longitude"] == pytest.approx(-58.399)

    def test_nonexistent_intersection_returns_none(
        self, manager: NetworkGraphManager
    ) -> None:
        assert manager.get_intersection("no-such-id") is None


# ---------------------------------------------------------------------------
# Tests — get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_middle_node_has_two_neighbors(self, manager: NetworkGraphManager) -> None:
        """int-002 connects to both int-001 and int-003 (bidirectional)."""
        neighbors = manager.get_neighbors("int-002")
        assert sorted(neighbors) == ["int-001", "int-003"]

    def test_endpoint_node_has_one_neighbor(self, manager: NetworkGraphManager) -> None:
        """int-001 connects to int-002 (forward edge); reverse from int-002 also exists."""
        neighbors = manager.get_neighbors("int-001")
        assert "int-002" in neighbors

    def test_nonexistent_node_returns_empty(self, manager: NetworkGraphManager) -> None:
        assert manager.get_neighbors("no-such-id") == []


# ---------------------------------------------------------------------------
# Tests — Edge Attributes
# ---------------------------------------------------------------------------

class TestEdgeAttributes:
    def test_edge_distance(self, manager: NetworkGraphManager) -> None:
        assert manager.get_edge_distance("int-001", "int-002") == pytest.approx(350.0)

    def test_edge_distance_reverse(self, manager: NetworkGraphManager) -> None:
        """Reverse edge should also have the same distance (bidirectional)."""
        assert manager.get_edge_distance("int-002", "int-001") == pytest.approx(350.0)

    def test_edge_distance_no_edge_returns_zero(
        self, manager: NetworkGraphManager
    ) -> None:
        assert manager.get_edge_distance("int-001", "int-003") == pytest.approx(0.0)

    def test_edge_distance_unknown_node_returns_zero(
        self, manager: NetworkGraphManager
    ) -> None:
        assert manager.get_edge_distance("int-001", "no-such-id") == pytest.approx(0.0)

    def test_edge_speed_limit(self, manager: NetworkGraphManager) -> None:
        assert manager.get_edge_speed_limit("int-001", "int-002") == pytest.approx(50.0)

    def test_edge_speed_limit_default_on_missing(
        self, manager: NetworkGraphManager
    ) -> None:
        """Non-existent edge returns the default speed limit of 50.0."""
        assert manager.get_edge_speed_limit("int-001", "no-such-id") == pytest.approx(
            50.0
        )


# ---------------------------------------------------------------------------
# Tests — Shortest Path
# ---------------------------------------------------------------------------

class TestShortestPath:
    def test_adjacent_path(self, manager: NetworkGraphManager) -> None:
        path = manager.shortest_path("int-001", "int-002")
        assert path == ["int-001", "int-002"]

    def test_multi_hop_path(self, manager: NetworkGraphManager) -> None:
        path = manager.shortest_path("int-001", "int-003")
        assert path == ["int-001", "int-002", "int-003"]

    def test_reverse_path_bidirectional(self, manager: NetworkGraphManager) -> None:
        path = manager.shortest_path("int-003", "int-001")
        assert path == ["int-003", "int-002", "int-001"]

    def test_no_path_returns_empty(self, tmp_path: Path) -> None:
        """Two disconnected nodes → empty path."""
        topo = _make_topology(
            intersections=[
                {
                    "id": "A",
                    "name": "A",
                    "location": {"latitude": 0, "longitude": 0},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
                {
                    "id": "B",
                    "name": "B",
                    "location": {"latitude": 1, "longitude": 1},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
            ],
            roads=[],
        )
        f = tmp_path / "disconnected.json"
        f.write_text(json.dumps(topo), encoding="utf-8")
        mgr = NetworkGraphManager(f)
        assert mgr.shortest_path("A", "B") == []

    def test_node_not_found_returns_empty(self, manager: NetworkGraphManager) -> None:
        assert manager.shortest_path("int-001", "nonexistent") == []

    def test_same_source_and_target(self, manager: NetworkGraphManager) -> None:
        path = manager.shortest_path("int-001", "int-001")
        assert path == ["int-001"]


# ---------------------------------------------------------------------------
# Tests — Shortest Path Length
# ---------------------------------------------------------------------------

class TestShortestPathLength:
    def test_adjacent_length(self, manager: NetworkGraphManager) -> None:
        length = manager.shortest_path_length("int-001", "int-002")
        assert length == pytest.approx(350.0)

    def test_multi_hop_length(self, manager: NetworkGraphManager) -> None:
        length = manager.shortest_path_length("int-001", "int-003")
        assert length == pytest.approx(700.0)

    def test_no_path_returns_inf(self, tmp_path: Path) -> None:
        topo = _make_topology(
            intersections=[
                {
                    "id": "X",
                    "name": "X",
                    "location": {"latitude": 0, "longitude": 0},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
                {
                    "id": "Y",
                    "name": "Y",
                    "location": {"latitude": 1, "longitude": 1},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
            ],
            roads=[],
        )
        f = tmp_path / "no_path.json"
        f.write_text(json.dumps(topo), encoding="utf-8")
        mgr = NetworkGraphManager(f)
        assert mgr.shortest_path_length("X", "Y") == float("inf")

    def test_node_not_found_returns_inf(self, manager: NetworkGraphManager) -> None:
        assert manager.shortest_path_length("int-001", "nonexistent") == float("inf")

    def test_same_node_length_is_zero(self, manager: NetworkGraphManager) -> None:
        assert manager.shortest_path_length("int-001", "int-001") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — Bidirectional Edges
# ---------------------------------------------------------------------------

class TestBidirectionalEdges:
    def test_forward_edge_exists(self, manager: NetworkGraphManager) -> None:
        assert manager.graph.has_edge("int-001", "int-002")

    def test_reverse_edge_exists(self, manager: NetworkGraphManager) -> None:
        assert manager.graph.has_edge("int-002", "int-001")

    def test_one_way_creates_single_edge(self, tmp_path: Path) -> None:
        topo = _make_topology(
            roads=[
                {
                    "id": "road-ow",
                    "source_id": "int-001",
                    "target_id": "int-002",
                    "distance_m": 200.0,
                    "speed_limit_kmh": 40.0,
                    "lanes": 2,
                    "one_way": True,
                    "road_class": "secondary",
                },
            ],
        )
        f = tmp_path / "oneway.json"
        f.write_text(json.dumps(topo), encoding="utf-8")
        mgr = NetworkGraphManager(f)
        assert mgr.graph.has_edge("int-001", "int-002")
        assert not mgr.graph.has_edge("int-002", "int-001")

    def test_one_way_no_reverse_path(self, tmp_path: Path) -> None:
        """Cannot traverse a one-way road in reverse."""
        topo = _make_topology(
            intersections=[
                {
                    "id": "int-001",
                    "name": "A",
                    "location": {"latitude": 0, "longitude": 0},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
                {
                    "id": "int-002",
                    "name": "B",
                    "location": {"latitude": 1, "longitude": 1},
                    "num_approaches": 2,
                    "has_signal": True,
                    "zone_id": "z",
                },
            ],
            roads=[
                {
                    "id": "road-ow",
                    "source_id": "int-001",
                    "target_id": "int-002",
                    "distance_m": 200.0,
                    "speed_limit_kmh": 40.0,
                    "lanes": 2,
                    "one_way": True,
                    "road_class": "secondary",
                },
            ],
        )
        f = tmp_path / "oneway_nopath.json"
        f.write_text(json.dumps(topo), encoding="utf-8")
        mgr = NetworkGraphManager(f)
        assert mgr.shortest_path("int-002", "int-001") == []


# ---------------------------------------------------------------------------
# Tests — Missing / Invalid Topology File
# ---------------------------------------------------------------------------

class TestMissingTopology:
    def test_missing_file_produces_empty_graph(self, tmp_path: Path) -> None:
        mgr = NetworkGraphManager(tmp_path / "nonexistent.json")
        assert mgr.intersection_count == 0
        assert mgr.road_count == 0
        assert mgr.graph.number_of_nodes() == 0

    def test_missing_file_queries_still_safe(self, tmp_path: Path) -> None:
        mgr = NetworkGraphManager(tmp_path / "nonexistent.json")
        assert mgr.get_neighbors("any") == []
        assert mgr.get_intersection("any") is None
        assert mgr.get_edge_distance("a", "b") == pytest.approx(0.0)
        assert mgr.get_edge_speed_limit("a", "b") == pytest.approx(50.0)
        assert mgr.shortest_path("a", "b") == []
        assert mgr.shortest_path_length("a", "b") == float("inf")
        assert mgr.get_all_intersection_ids() == []


# ---------------------------------------------------------------------------
# Tests — Corridor Identification
# ---------------------------------------------------------------------------

class TestCorridorIdentification:
    def test_identifies_at_least_one_corridor(
        self, manager: NetworkGraphManager
    ) -> None:
        corridors = manager.identify_corridors()
        assert len(corridors) >= 1

    def test_corridor_contains_all_nodes(
        self, manager: NetworkGraphManager
    ) -> None:
        """The linear 3-node network should produce a corridor through all nodes."""
        corridors = manager.identify_corridors()
        all_nodes_in_corridors = {n for c in corridors for n in c}
        assert "int-001" in all_nodes_in_corridors
        assert "int-002" in all_nodes_in_corridors
        assert "int-003" in all_nodes_in_corridors

    def test_corridor_is_ordered_path(self, manager: NetworkGraphManager) -> None:
        """Each corridor should be a valid ordered path through the graph."""
        corridors = manager.identify_corridors()
        for corridor in corridors:
            assert len(corridor) >= 2
            # Verify consecutive nodes are connected
            for i in range(len(corridor) - 1):
                assert manager.graph.has_edge(corridor[i], corridor[i + 1])

    def test_empty_graph_no_corridors(self, tmp_path: Path) -> None:
        topo = _make_topology(intersections=[], roads=[])
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(topo), encoding="utf-8")
        mgr = NetworkGraphManager(f)
        assert mgr.identify_corridors() == []


# ---------------------------------------------------------------------------
# Tests — Graph Property (direct access)
# ---------------------------------------------------------------------------

class TestGraphProperty:
    def test_graph_is_digraph(self, manager: NetworkGraphManager) -> None:
        import networkx as nx

        assert isinstance(manager.graph, nx.DiGraph)

    def test_node_attributes_stored(self, manager: NetworkGraphManager) -> None:
        node_data = manager.graph.nodes["int-001"]
        assert node_data["name"] == "Intersection A"
        assert node_data["zone_id"] == "zone-a"
        assert node_data["lat"] == pytest.approx(-34.600)
        assert node_data["lng"] == pytest.approx(-58.400)

    def test_edge_attributes_stored(self, manager: NetworkGraphManager) -> None:
        edge_data = manager.graph["int-001"]["int-002"]
        assert edge_data["distance_m"] == pytest.approx(350.0)
        assert edge_data["speed_limit_kmh"] == pytest.approx(50.0)
        assert edge_data["lanes"] == 3
        assert edge_data["road_class"] == "primary"
