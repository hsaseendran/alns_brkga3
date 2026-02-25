# ALNS-BRKGA Hybrid Solver

A GPU-accelerated combinatorial optimization solver that combines two metaheuristic frameworks for high-quality solutions:

1. **Phase 1 — ALNS** (Adaptive Large Neighborhood Search): Rapidly explores the solution space using destroy-and-repair operators on hundreds of parallel GPU threads
2. **Phase 2 — BRKGA** (Biased Random-Key Genetic Algorithm): Refines the ALNS solution through population-based evolutionary search with warm-start injection

The ALNS solution is converted to a BRKGA chromosome and injected into every population, giving BRKGA a strong starting point. BRKGA then evolves the population using crossover, mutation, and selection to further improve the solution.

## Supported Problems

| Problem | Description | Input Format |
|---------|-------------|--------------|
| **TSP** | Traveling Salesman Problem — find shortest tour visiting all cities | TSPLIB (.tsp) or CSV distance matrix |
| **CVRP** | Capacitated Vehicle Routing — minimize total distance with vehicle capacity constraints | TSPLIB CVRP (.vrp) |
| **VRP-RPD** | Vehicle Routing with Release & Pickup-Delivery — minimize makespan with drop/pick operations | CSV distance matrix + optional processing times |

## Architecture

```
                    ALNS-BRKGA Hybrid Pipeline
                    -------------------------

  Instance File ──> [ ALNS Phase ]  ──>  Solution  ──>  [ Convert to    ]
  (.tsp/.vrp/.csv)  Multi-GPU ALNS      (tour/routes)    [ BRKGA chromosome ]
                    Destroy & Repair                           │
                    256 threads/block                           v
                    N GPUs × 32 blocks              [ BRKGA Phase ]
                                                    Multi-GPU BRKGA
                                                    Evolve populations
                                                    N GPUs × K populations
                                                           │
                                                           v
                                                    Improved Solution
                                                    (terminal + JSON)
```

**CVRP Strategy**: ALNS solves a TSP relaxation (ignoring capacity) to find a good client ordering. BRKGA's greedy-split decoder then partitions the permutation into capacity-feasible routes.

**VRP-RPD Strategy**: ALNS solves the full VRP-RPD problem with cross-agent pickup and inventory constraints. The customer execution order is extracted and used to warm-start BRKGA's simplified greedy decoder.

## Prerequisites

- **CUDA Toolkit** 12.0+ with `nvcc`
- **CMake** 3.18+
- **Linux** (tested on Ubuntu 22.04)
- **NVIDIA GPU** — Ada Lovelace (L40S, RTX 4090) recommended; Ampere (A100) also supported
- **ALNS Framework** at `../ALNS/framework` (header-only, no separate build needed)
- **BRKGA3 Framework** at `../brkga_3/brkga3` (built automatically as a dependency)

## Building

```bash
cd alns_brkga
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The build produces a single executable: `alns_brkga`

To target a different GPU architecture (e.g., Ampere SM 80):
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

## Usage

```
./alns_brkga <problem> <instance> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--alns-time <sec>` | 30 | ALNS time limit in seconds |
| `--alns-gpus <n>` | -1 (auto) | Number of GPUs for ALNS |
| `--alns-spg <n>` | 32 | ALNS solutions per GPU |
| `--brkga-gens <n>` | 1000 | BRKGA generations |
| `--brkga-gpus <n>` | -1 (auto) | Number of GPUs for BRKGA |
| `--brkga-pops <n>` | 3 | BRKGA populations per GPU |
| `--brkga-pop-size <n>` | 1024 | BRKGA population size |
| `--seed <n>` | 42 | Random seed |
| `--output <file>` | — | Write JSON results to file |
| `--cold` | — | Skip ALNS, run BRKGA only (cold start) |
| `-q` / `--quiet` | — | Suppress progress output |

**VRP-RPD specific:**

| Option | Default | Description |
|--------|---------|-------------|
| `--agents <n>` | 5 | Number of agents/vehicles |
| `--resources <n>` | 2 | Resources per agent |
| `--proc-times <file>` | — | Processing times CSV file |

## Examples

### TSP — Traveling Salesman Problem

```bash
# Solve a280.tsp: ALNS for 30s, then BRKGA for 1000 generations
./alns_brkga tsp ../brkga_3/brkga3/data/tsp/tsplib/a280.tsp \
    --alns-time 30 --brkga-gens 1000

# Large instance with more BRKGA resources
./alns_brkga tsp ../brkga_3/brkga3/data/tsp/tsplib/lu980.tsp \
    --alns-time 60 --brkga-gens 2000 --brkga-pop-size 2048

# Cold start (BRKGA only, no ALNS)
./alns_brkga tsp ../brkga_3/brkga3/data/tsp/tsplib/a280.tsp --cold

# Save results to JSON
./alns_brkga tsp ../brkga_3/brkga3/data/tsp/tsplib/a280.tsp --output results.json
```

### CVRP — Capacitated Vehicle Routing

```bash
# Solve X-n101-k25.vrp
./alns_brkga cvrp ../brkga_3/brkga3/data/set-x/X-n101-k25.vrp \
    --alns-time 30 --brkga-gens 1000

# Larger instance with more compute
./alns_brkga cvrp ../brkga_3/brkga3/data/set-x/X-n1001-k43.vrp \
    --alns-time 120 --brkga-gens 5000 --brkga-pop-size 4096
```

### VRP-RPD — Vehicle Routing with Release & Pickup-Delivery

```bash
# Solve with 5 agents
./alns_brkga vrprpd ../ALNS/berlin52/distance_matrix/berlin52_matrix.csv \
    --agents 5 --alns-time 30

# With processing times and 8 agents
./alns_brkga vrprpd distances.csv \
    --proc-times processing_times.csv \
    --agents 8 --resources 3 --alns-time 60
```

## Output

### Terminal Output

```
============================================================
  ALNS-BRKGA Hybrid Solver
  Problem:  tsp
  Instance: a280.tsp
  Mode:     WARM START (ALNS -> BRKGA)
============================================================

--- Phase 1: ALNS (Adaptive Large Neighborhood Search) ---
  GPUs: 8 | Solutions/GPU: 32 | Time limit: 30s
  ALNS completed: objective = 2637.00, time = 30.1s

--- Phase 2: BRKGA (Biased Random-Key Genetic Algorithm) ---
  Warm-start injected: 2637.00
  GPUs: 8 | Populations: 24 | Pop size: 1024
  Gen   100 | Best:      2637.00 | Time: 0.852s
  Gen   500 | Best:      2621.00 | Time: 3.214s
  Gen  1000 | Best:      2614.00 | Time: 6.103s
  BRKGA completed: objective = 2614.00, time = 6.1s

============================================================
  RESULTS
  ALNS solution:     2637.00  (30.1s)
  BRKGA initial:     2637.00
  BRKGA final:       2614.00  (1000 gens, 6.1s)
  BRKGA improvement: 0.87%
  Final objective:   2614.00
  Total time:        36.2s
============================================================
```

### JSON Output

When `--output result.json` is specified:

```json
{
  "solver": "alns_brkga",
  "version": "1.0",
  "problem": "tsp",
  "instance": "a280.tsp",
  "alns": {
    "objective": 2637.0,
    "time_seconds": 30.1
  },
  "brkga": {
    "warm_start_objective": 2637.0,
    "final_objective": 2614.0,
    "generations": 1000,
    "time_seconds": 6.1
  },
  "result": {
    "objective": 2614.0,
    "total_time_seconds": 36.2,
    "improvement_pct": 0.87
  },
  "solution": {
    "total_distance": 2614.0,
    "num_cities": 280,
    "tour": [0, 3, 12, ...]
  }
}
```

## Algorithm Details

### ALNS (Adaptive Large Neighborhood Search)

ALNS iteratively improves solutions by destroying part of the current solution and repairing it:

- **Destroy operators**: Random removal, worst removal, Shaw (relatedness) removal, cluster removal
- **Repair operators**: Greedy insertion, regret-k insertion, random insertion
- **Acceptance**: Simulated annealing with adaptive reheating
- **Parallelism**: Each GPU block runs an independent ALNS instance (256 threads cooperate within a block)
- **Communication**: Best solutions are shared across all GPU blocks via managed memory

### BRKGA (Biased Random-Key Genetic Algorithm)

BRKGA evolves a population of random-key chromosomes (float arrays in (0,1]):

- **Encoding**: Random keys are sorted to produce permutations (for TSP/CVRP/VRP-RPD)
- **Decoding**: Problem-specific GPU decoder evaluates each chromosome
- **Selection**: Elites survive, crossover biased toward elite parents, mutants add diversity
- **Island model**: Multiple independent populations with periodic migration of elite chromosomes
- **Warm-start**: ALNS solution injected as chromosome slot 0 in every population

### Why Hybrid?

ALNS excels at rapid exploration through large neighborhood moves — it can quickly find good solutions by removing and reinserting multiple elements at once. However, it can get trapped in local optima.

BRKGA provides fine-grained refinement through genetic crossover. The population-based approach maintains diversity, and the biased crossover gradually combines good building blocks from different solutions. Starting from a strong ALNS solution, BRKGA can polish the solution quality further.

## References

- Ropke, S., & Pisinger, D. (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*.
- Gonçalves, J. F., & Resende, M. G. C. (2011). Biased random-key genetic algorithms for combinatorial optimization. *Journal of Heuristics*.
- Andrade, C. E., et al. (2021). The Multi-Parent Biased Random-Key Genetic Algorithm with Implicit Path-Relinking. *European Journal of Operational Research*.
