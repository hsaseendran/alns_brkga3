#!/usr/bin/env python3
"""
VRP-RPD Solution Validator for ALNS-BRKGA Hybrid Solver

Validates solutions for the Vehicle Routing with Release & Pickup-Delivery problem.

Validates:
1. All customers served (exactly one dropoff and one pickup per customer)
2. Resource constraints (max k resources deployed per agent at any time)
3. Processing time constraints (pickup >= dropoff + proc_time)
4. Cross-agent pickup validity (pickup agent may differ from dropoff agent)
5. Makespan calculation (max agent completion time including return to depot)
6. Travel time consistency (arrival times respect distances)

Supports two JSON formats:
  - ALNS-BRKGA output:  {"solver": ..., "solution": {"problem": ..., "routes": ...}}
  - Standalone:         {"problem": ..., "makespan": ..., "routes": ...}

Both route formats are accepted:
  - v1: "operations": [{"customer": N, "type": "D", "time": T}]
  - v2: "stops":      [{"node": N, "op": "D", "time": T}]

Usage:
    python validate_vrprpd.py <solution.json> <proc_times.csv> --matrix <distance_matrix.csv>
    python validate_vrprpd.py <solution.json> <proc_times.csv> --tsp <file.tsp>
"""

import json
import math
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Coordinate:
    x: float
    y: float


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


# ============================================================================
# File loaders
# ============================================================================

def load_csv_distance_matrix(filename: str) -> List[List[float]]:
    """Load distance matrix from CSV file (no header, comma-separated)."""
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                row = [float(x) for x in line.split(',')]
                matrix.append(row)
    return matrix


def parse_tsp_file(filename: str) -> Tuple[List[Coordinate], str, str]:
    """Parse TSPLIB format TSP file and return coordinates and edge weight type."""
    coords = []
    instance_name = ""
    edge_weight_type = "EUC_2D"
    in_coord_section = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('NAME:'):
                instance_name = line.split(':')[1].strip()
            elif line.startswith('EDGE_WEIGHT_TYPE:'):
                edge_weight_type = line.split(':')[1].strip()
            elif line == 'NODE_COORD_SECTION':
                in_coord_section = True
            elif line == 'EOF':
                break
            elif in_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    while len(coords) < idx:
                        coords.append(None)
                    coords[idx - 1] = Coordinate(x, y)

    return coords, instance_name, edge_weight_type


def build_distance_matrix(coords: List[Coordinate], edge_weight_type: str = "EUC_2D") -> List[List[float]]:
    """Build distance matrix from coordinates."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]

    def euc_2d(c1, c2):
        return round(math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2))

    def ceil_2d(c1, c2):
        return math.ceil(math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2))

    def geo(c1, c2):
        PI = 3.141592653589793
        RRR = 6378.388
        def to_rad(coord):
            deg = int(coord)
            return PI * (deg + 5.0 * (coord - deg) / 3.0) / 180.0
        q1 = math.cos(to_rad(c1.y) - to_rad(c2.y))
        q2 = math.cos(to_rad(c1.x) - to_rad(c2.x))
        q3 = math.cos(to_rad(c1.x) + to_rad(c2.x))
        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    def att(c1, c2):
        dx, dy = c1.x - c2.x, c1.y - c2.y
        rij = math.sqrt((dx*dx + dy*dy) / 10.0)
        tij = int(round(rij))
        return tij + 1 if tij < rij else tij

    dist_func = {
        "EUC_2D": euc_2d, "CEIL_2D": ceil_2d, "GEO": geo, "ATT": att
    }.get(edge_weight_type, lambda c1, c2: math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = dist_func(coords[i], coords[j])
    return dist


def parse_proc_times(filename: str) -> List[float]:
    """Parse processing times file (plain text or CSV with header)."""
    proc_times = []
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return proc_times
    if ',' in lines[0]:
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 2:
                proc_times.append(float(parts[1]))
    else:
        for line in lines:
            proc_times.append(float(line))
    return proc_times


# ============================================================================
# Solution parser (handles both ALNS-BRKGA and standalone formats)
# ============================================================================

def normalize_solution(raw: dict) -> dict:
    """
    Normalize a solution JSON to a canonical format:
      {
        "problem": {"n_customers": N, "n_agents": M, "resources_per_agent": K},
        "makespan": float,
        "routes": [
          {"agent": int, "finish_time": float,
           "stops": [{"node": int, "op": "D"/"P", "time": float}]}
        ]
      }
    """
    # If alns_brkga wrapper, extract the "solution" sub-object
    if "solver" in raw and "solution" in raw:
        sol = raw["solution"]
    else:
        sol = raw

    # Normalize problem info
    problem = sol.get("problem", sol)
    n_cust = problem.get("n_customers", problem.get("num_customers"))
    n_agents = problem.get("n_agents", problem.get("num_agents"))
    resources = problem.get("resources_per_agent", 2)

    makespan = sol.get("makespan", 0)

    # Normalize routes
    routes = []
    for route in sol.get("routes", []):
        agent = route.get("agent", -1)
        finish_time = route.get("finish_time", route.get("completion_time", 0))

        # Accept either "stops" (v2) or "operations" (v1)
        raw_stops = route.get("stops", route.get("operations", []))
        stops = []
        for s in raw_stops:
            node = s.get("node", s.get("customer"))
            op = s.get("op", s.get("type"))
            time = s.get("time")
            stops.append({"node": node, "op": op, "time": time})

        routes.append({
            "agent": agent,
            "finish_time": finish_time,
            "stops": stops
        })

    return {
        "problem": {
            "n_customers": n_cust,
            "n_agents": n_agents,
            "resources_per_agent": resources
        },
        "makespan": makespan,
        "routes": routes
    }


# ============================================================================
# Resource constraint simulation
# ============================================================================

def simulate_resource_constraints(solution: dict, verbose: bool = False):
    """
    Event-based simulation to check physical inventory resource constraints.

    Each agent starts with k resources in hand:
      - Dropoff: agent uses 1 resource (must have >= 1)
      - Pickup: agent gains 1 resource
      - Constraint: 0 <= inventory <= k at all times
    """
    errors = []
    n_agents = solution['problem']['n_agents']
    resources_per_agent = solution['problem']['resources_per_agent']

    events = []
    for route in solution['routes']:
        agent = route['agent']
        for stop in route['stops']:
            if stop['node'] == 0:
                continue
            events.append((stop['time'], stop['op'], agent, stop['node']))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'D' else 1))

    inventory = {a: resources_per_agent for a in range(n_agents)}
    max_inventory = {a: resources_per_agent for a in range(n_agents)}

    for time, op, agent, customer in events:
        if op == 'D':
            if inventory[agent] <= 0:
                errors.append(
                    f"Agent {agent}: Cannot dropoff at customer {customer} at time {time:.2f}! "
                    f"Has {inventory[agent]} resources (need >= 1)")
            inventory[agent] -= 1
        elif op == 'P':
            inventory[agent] += 1
            max_inventory[agent] = max(max_inventory[agent], inventory[agent])
            if inventory[agent] > resources_per_agent:
                errors.append(
                    f"Agent {agent}: Exceeded capacity at time {time:.2f}! "
                    f"Has {inventory[agent]} resources (max: {resources_per_agent})")

        if verbose:
            inv_str = [inventory.get(a, 0) for a in range(n_agents)]
            print(f"  t={time:.0f}: {op} node {customer} by agent {agent}, inventory: {inv_str}")

    return errors, max_inventory


# ============================================================================
# Main validator
# ============================================================================

def validate_solution(solution: dict, proc_times: List[float],
                      dist: List[List[float]], verbose: bool = False,
                      travel_tolerance: float = 1.0) -> ValidationResult:
    """
    Validate a VRP-RPD solution.

    Checks:
    1. All customers served (exactly one D and one P per customer)
    2. Resource constraints (max k in hand per agent)
    3. Processing time constraints (pickup >= dropoff + proc_time)
    4. Travel time consistency
    5. Makespan calculation
    """
    errors = []
    warnings = []

    n_customers = solution['problem']['n_customers']
    n_agents = solution['problem']['n_agents']
    resources_per_agent = solution['problem']['resources_per_agent']
    reported_makespan = solution['makespan']

    # --- 1. Collect all operations globally ---
    dropoffs = {}   # node -> (agent, time)
    pickups = {}    # node -> (agent, time)

    for route in solution['routes']:
        agent = route['agent']
        for stop in route['stops']:
            node = stop['node']
            op = stop['op']
            time = stop['time']

            if node == 0:
                continue

            if op == 'D':
                if node in dropoffs:
                    errors.append(f"Customer {node}: Multiple dropoffs!")
                dropoffs[node] = (agent, time)
            elif op == 'P':
                if node in pickups:
                    errors.append(f"Customer {node}: Multiple pickups!")
                pickups[node] = (agent, time)

    # --- 2. Resource constraints ---
    resource_errors, max_inv = simulate_resource_constraints(solution, verbose)
    errors.extend(resource_errors)

    # --- 3. Travel time validation and finish times ---
    agent_finish_times = []

    for route in solution['routes']:
        agent = route['agent']
        stops = route['stops']
        reported_finish = route['finish_time']

        current_time = 0.0
        current_node = 0  # depot

        for stop in stops:
            node = stop['node']
            time = stop['time']

            if node == 0:
                continue

            expected_arrival = current_time + dist[current_node][node]
            tol = max(travel_tolerance, abs(expected_arrival) * 2e-7)
            if time < expected_arrival - tol:
                errors.append(
                    f"Agent {agent}: Arrived at node {node} at t={time:.2f} "
                    f"but travel from node {current_node} takes {dist[current_node][node]:.2f} "
                    f"(earliest: {expected_arrival:.2f}, gap: {expected_arrival - time:.2f})")

            current_time = time
            current_node = node

        # Return to depot
        if stops:
            last_node = stops[-1]['node']
            last_time = stops[-1]['time']
            calc_finish = last_time + dist[last_node][0]
        else:
            calc_finish = 0.0

        agent_finish_times.append((agent, calc_finish, reported_finish))

        finish_tol = max(1.0, abs(calc_finish) * 2e-7)
        if abs(calc_finish - reported_finish) > finish_tol:
            warnings.append(
                f"Agent {agent}: Reported finish {reported_finish:.2f} vs "
                f"calculated {calc_finish:.2f}")

    # --- 4. All customers served ---
    for cust in range(1, n_customers + 1):
        if cust not in dropoffs:
            errors.append(f"Customer {cust}: No dropoff found!")
        if cust not in pickups:
            errors.append(f"Customer {cust}: No pickup found!")

    for cust in dropoffs:
        if cust < 1 or cust > n_customers:
            errors.append(f"Invalid customer {cust} in dropoffs (valid: 1-{n_customers})")
    for cust in pickups:
        if cust < 1 or cust > n_customers:
            errors.append(f"Invalid customer {cust} in pickups (valid: 1-{n_customers})")

    # --- 5. Processing time constraints ---
    cross_agent_pickups = []
    for cust in range(1, n_customers + 1):
        if cust in dropoffs and cust in pickups:
            d_agent, d_time = dropoffs[cust]
            p_agent, p_time = pickups[cust]
            proc_time = proc_times[cust - 1]

            min_pickup = d_time + proc_time
            tol = max(travel_tolerance, abs(min_pickup) * 2e-7)
            if p_time < min_pickup - tol:
                errors.append(
                    f"Customer {cust}: Pickup at {p_time:.2f} before processing done! "
                    f"(drop: {d_time:.2f}, proc: {proc_time}, min pickup: {min_pickup:.2f})")

            if d_agent != p_agent:
                cross_agent_pickups.append(
                    (cust, d_agent, p_agent, d_time, p_time, proc_time))
                if verbose:
                    print(f"  Cross-agent: Customer {cust}: D by Agent {d_agent} @ {d_time:.2f}, "
                          f"P by Agent {p_agent} @ {p_time:.2f} (proc: {proc_time})")

    # --- 6. Makespan ---
    calculated_makespan = 0.0
    bottleneck_agent = -1
    for agent, calc_finish, _ in agent_finish_times:
        if calc_finish > calculated_makespan:
            calculated_makespan = calc_finish
            bottleneck_agent = agent

    ms_tol = max(1.0, abs(calculated_makespan) * 2e-7)
    if abs(reported_makespan - calculated_makespan) > ms_tol:
        warnings.append(
            f"Reported makespan {reported_makespan:.2f} vs "
            f"calculated {calculated_makespan:.2f} "
            f"(diff: {abs(reported_makespan - calculated_makespan):.2f})")

    stats = {
        'n_customers': n_customers,
        'n_agents': n_agents,
        'resources_per_agent': resources_per_agent,
        'calculated_makespan': calculated_makespan,
        'reported_makespan': reported_makespan,
        'bottleneck_agent': bottleneck_agent,
        'cross_agent_pickups': len(cross_agent_pickups),
        'max_inventory': max_inv,
        'agent_finish_times': {a: f for a, f, _ in agent_finish_times},
        'customers_served': len(set(dropoffs.keys()) & set(pickups.keys())),
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


# ============================================================================
# Report printer
# ============================================================================

def print_report(result: ValidationResult, dist_type: str = "CSV"):
    print()
    print("=" * 60)
    print("       VRP-RPD SOLUTION VALIDATION REPORT")
    print("=" * 60)

    status = "VALID" if result.is_valid else "INVALID"
    print(f"\nStatus: {status}")

    s = result.stats
    print(f"\nProblem: {s['n_customers']} customers, "
          f"{s['n_agents']} agents, "
          f"{s['resources_per_agent']} resources/agent")
    print(f"Distance type: {dist_type}")

    print(f"\nCustomers served: {s['customers_served']}/{s['n_customers']}")

    print(f"\nMakespan:")
    print(f"  Reported:   {s['reported_makespan']:.2f}")
    print(f"  Calculated: {s['calculated_makespan']:.2f}")
    print(f"  Bottleneck: Agent {s['bottleneck_agent']}")

    print(f"\nCross-agent pickups: {s['cross_agent_pickups']}")

    print(f"\nAgent finish times:")
    for agent, finish_time in sorted(s['agent_finish_times'].items()):
        max_inv = s['max_inventory'].get(agent, 0)
        marker = " <-- BOTTLENECK" if agent == s['bottleneck_agent'] else ""
        print(f"  Agent {agent}: {finish_time:>10.2f}  (max inventory: {max_inv}){marker}")

    if result.errors:
        print(f"\n{'!' * 60}")
        print(f"ERRORS ({len(result.errors)}):")
        print('!' * 60)
        for err in result.errors:
            print(f"  [X] {err}")

    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for warn in result.warnings:
            print(f"  [!] {warn}")

    if result.is_valid and not result.warnings:
        print(f"\nAll constraints satisfied.")

    print("\n" + "=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate VRP-RPD solutions from ALNS-BRKGA solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using CSV distance matrix:
    python validate_vrprpd.py result.json proc_times.csv \\
        --matrix distance_matrix.csv

    # Using TSP file:
    python validate_vrprpd.py result.json proc_times.csv \\
        --tsp instance.tsp

    # Verbose (shows every event):
    python validate_vrprpd.py result.json proc_times.csv \\
        --matrix distance_matrix.csv -v
        """)
    parser.add_argument('solution', help='Solution JSON file')
    parser.add_argument('proc_times', help='Processing times file (txt or csv)')
    parser.add_argument('--matrix', '-m', type=str, default=None,
                        help='CSV distance matrix file')
    parser.add_argument('--tsp', type=str, default=None,
                        help='TSP file with coordinates')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output with event trace')
    parser.add_argument('--tolerance', '-t', type=float, default=1.0,
                        help='Travel time tolerance (default: 1.0)')

    args = parser.parse_args()

    try:
        # Load distance matrix
        dist = None
        dist_type = "CSV"

        if args.matrix:
            print(f"Loading distance matrix: {args.matrix}")
            dist = load_csv_distance_matrix(args.matrix)
            print(f"  Size: {len(dist)} x {len(dist[0])}")
        elif args.tsp:
            print(f"Loading TSP file: {args.tsp}")
            coords, name, ewt = parse_tsp_file(args.tsp)
            print(f"  Instance: {name}, {len(coords)} nodes, type: {ewt}")
            dist = build_distance_matrix(coords, ewt)
            dist_type = ewt
        else:
            print("Error: Provide either --matrix or --tsp")
            return 1

        # Load processing times
        print(f"Loading processing times: {args.proc_times}")
        proc_times = parse_proc_times(args.proc_times)
        print(f"  {len(proc_times)} processing times loaded")

        # Load and normalize solution
        print(f"Loading solution: {args.solution}")
        with open(args.solution, 'r') as f:
            raw = json.load(f)
        solution = normalize_solution(raw)
        print(f"  {solution['problem']['n_customers']} customers, "
              f"{solution['problem']['n_agents']} agents")

        # Validate
        result = validate_solution(
            solution, proc_times, dist,
            verbose=args.verbose,
            travel_tolerance=args.tolerance)

        # Report
        print_report(result, dist_type)

        return 0 if result.is_valid else 1

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
