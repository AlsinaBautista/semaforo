"""
Generate a 3-intersection corridor network AND proper corridor routes for SUMO.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BRAIN_DIR))

from sumo_networks.network_generator import NetworkGenerator

def generate_corridor_routes(output_dir: str, num_intersections: int = 3, sim_end: float = 3600.0):
    """Generate routes that actually traverse the corridor end-to-end."""
    out = Path(output_dir)
    rou_path = out / "corridor.rou.xml"
    
    last_i = num_intersections - 1
    
    with open(rou_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n\n')
        
        # Vehicle types
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                'length="5.0" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>\n')
        f.write('    <vType id="bus" accel="1.5" decel="3.5" sigma="0.5" '
                'length="12.0" minGap="3.0" maxSpeed="11.11" guiShape="bus" color="0,128,255"/>\n\n')
        
        # --- Main corridor routes (East-West) ---
        # West → East: west_entry_to_c0 → c0_to_c1 → c1_to_c2 → c2_to_east_entry
        ew_edges = ["west_entry_to_c0"]
        for i in range(num_intersections - 1):
            ew_edges.append(f"c{i}_to_c{i+1}")
        ew_edges.append(f"c{last_i}_to_east_entry")
        
        # East → West: east_entry_to_cN → ... → c0_to_west_entry
        we_edges = [f"east_entry_to_c{last_i}"]
        for i in range(last_i, 0, -1):
            we_edges.append(f"c{i}_to_c{i-1}")
        we_edges.append("c0_to_west_entry")
        
        f.write(f'    <route id="ew_through" edges="{" ".join(ew_edges)}"/>\n')
        f.write(f'    <route id="we_through" edges="{" ".join(we_edges)}"/>\n\n')
        
        # --- Cross-street routes (North-South) for each intersection ---
        for i in range(num_intersections):
            ns_edges = f"c{i}_north_in c{i}_south_out"
            sn_edges = f"c{i}_south_in c{i}_north_out"
            f.write(f'    <route id="ns_c{i}" edges="{ns_edges}"/>\n')
            f.write(f'    <route id="sn_c{i}" edges="{sn_edges}"/>\n')
        f.write('\n')
        
        # --- Partial corridor routes (vehicles entering/exiting mid-corridor) ---
        # West → exit at c1 north
        f.write(f'    <route id="w_to_c1n" edges="west_entry_to_c0 c0_to_c1 c1_north_out"/>\n')
        # South c0 → East  
        f.write(f'    <route id="c0s_to_e" edges="c0_south_in c0_to_c1 c1_to_c2 c{last_i}_to_east_entry"/>\n\n')
        
        # --- Traffic flows ---
        # Heavy main avenue traffic (1200 veh/h each direction)
        prob_ew = round(1200 / 3600, 4)
        prob_we = round(1200 / 3600, 4)
        f.write(f'    <flow id="flow_ew" type="car" route="ew_through" '
                f'begin="0" end="{sim_end}" probability="{prob_ew}" '
                f'departLane="best" departSpeed="max"/>\n')
        f.write(f'    <flow id="flow_we" type="car" route="we_through" '
                f'begin="0" end="{sim_end}" probability="{prob_we}" '
                f'departLane="best" departSpeed="max"/>\n\n')
        
        # Moderate cross-street traffic (300 veh/h each direction per intersection)
        prob_cross = round(300 / 3600, 4)
        for i in range(num_intersections):
            f.write(f'    <flow id="flow_ns_c{i}" type="car" route="ns_c{i}" '
                    f'begin="0" end="{sim_end}" probability="{prob_cross}" '
                    f'departLane="best" departSpeed="max"/>\n')
            f.write(f'    <flow id="flow_sn_c{i}" type="car" route="sn_c{i}" '
                    f'begin="0" end="{sim_end}" probability="{prob_cross}" '
                    f'departLane="best" departSpeed="max"/>\n')
        f.write('\n')
        
        # Some buses on the main avenue (60 veh/h)
        prob_bus = round(60 / 3600, 4)
        f.write(f'    <flow id="flow_bus_ew" type="bus" route="ew_through" '
                f'begin="0" end="{sim_end}" probability="{prob_bus}" '
                f'departLane="best" departSpeed="max"/>\n')
        f.write(f'    <flow id="flow_bus_we" type="bus" route="we_through" '
                f'begin="0" end="{sim_end}" probability="{prob_bus}" '
                f'departLane="best" departSpeed="max"/>\n\n')
        
        # Partial routes for realism
        f.write(f'    <flow id="flow_w_c1n" type="car" route="w_to_c1n" '
                f'begin="0" end="{sim_end}" probability="0.02" '
                f'departLane="best" departSpeed="max"/>\n')
        f.write(f'    <flow id="flow_c0s_e" type="car" route="c0s_to_e" '
                f'begin="0" end="{sim_end}" probability="0.02" '
                f'departLane="best" departSpeed="max"/>\n\n')
        
        f.write('</routes>\n')
    
    print(f"  ✓ Corridor routes written → {rou_path}")
    return str(rou_path)


def main():
    print("🚦 Generating 3-intersection corridor for Green Wave simulation...")
    gen = NetworkGenerator()
    
    output_dir = BRAIN_DIR / "sumo_networks" / "corridor"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the network
    net_path = gen.generate_corridor(
        output_dir=str(output_dir),
        num_intersections=3,
        spacing_m=350.0,
        num_lanes=3,
        speed_kmh=50.0,
        approach_length=200.0,
    )
    print(f"  ✓ Network compiled → {net_path}")
    
    # Generate proper corridor routes
    rou_path = generate_corridor_routes(str(output_dir), num_intersections=3)
    
    print(f"\n✅ All files ready in {output_dir}/")
    print(f"   Network: corridor.net.xml")
    print(f"   Routes:  corridor.rou.xml")


if __name__ == "__main__":
    main()
