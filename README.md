# Prime - Decentralized Training At Scale
Prime (previously called ZeroBand) is a framework for efficient, globally distributed training of AI models over the internet.

https://github.com/user-attachments/assets/c034d2a2-400c-4bf8-acd0-c84b6c897d69

## Key Features
- **Fault Tolerant Training** with Dynamic On-/Off Ramping of Workers. Prime introduces the `ElasticDeviceMesh` concept, which provides:
    - Dynamic global process groups for communication via the internet
    - Fixed local process groups for distributed training inside one node / datacenter
    - The local process groups are fixed, while the global process groups can be recreated dynamically
    - **Heartbeat / Deathrattle mechanism:** Each node signals that it’s still online by sending a heartbeat to the global TCP store. If it dies it sends a deathrattle. If it dies before being able to send a deathrattle the node will be ejected before the outer step communication.
    - **World Resolution:** Dynamically manages handling of joiners and leavers, remapping ranks, and triggering global process group reinitializations.
    - **Independent TCP Stores:** Contrary to usual PyTorch distributed training we leverage multiples TCPStores. For instance, each DiLoCo node has its own TCPStore for local communication.
- Live Checkpoint Recovery
- Async Checkpointing
- Custom Int8 All-Reduce Kernels
    - Python compression is slow -> C++ 
    - Quantize aware ring reduce
- [DiLoCo](https://arxiv.org/abs/2311.08105) implementation
    - Our DiLoCo optimizers are shared allowing us to open multiple connections at the same time and maximising bandwidth utilization
- FSDP2 based

A research paper explaining the Prime framework is coming soon.

## Getting Started

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

2. Set up the environment:
```bash
uv venv
source .venv/bin/activate
uv sync --extra all
uv pip install flash-attn --no-build-isolation
git submodule update --init --recursive
```

### Quick Check

Verify your setup:

```bash
ZERO_BAND_LOG_LEVEL=DEBUG torchrun --nproc_per_node=2 src/zeroband/train.py @configs/debug/normal.toml
```

## Usage

### Running DiLoCo

To test DiLoCo locally you can use the helper script `scripts/simulatsimulate_multi_nodee_mutl.sh` 

```bash
# Using 4 GPUs
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 2 src/zeroband/train.py @configs/debug/diloco.toml

# Using 2 GPUs
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 1 src/zeroband/train.py @configs/debug/diloco.toml
```

> **Note:** Single GPU setups are currently not supported due to an FSDP implementation bug.

### Running Tests

Ensure you have at least two GPU to run the full test suite:
```bash
uv run pytest
```

### On/Off Ramping Routines
- For the first initialisation, all the GLOBAL env vars matter and will be used by the nodes to initialize.
- When nodes join, only the `GLOBAL_ADDR` and `GLOBAL_PORT` matter. You still have to set `GLOBAL_RANK` and `GLOBAL_WORLD_SIZE` but they will be updated when the global pg initializes.
- When a node wishes to offboard, it must call `edm._queue_leave()` and then `edm.maybe_reinit_global_pg()`. The mechanism is that it has to tell the master it is leaving and then join the the next `edm.maybe_reinit_global_pg()` in order to not deadlock the barrier for master's `_resolve_world()`. Jackmin is trying to change this behavior such that the leaving node can leave without having to `edm.maybe_reinit_global_pg()` or without `edm._queue_leave()` but they are required for now.

## Environment variables
### Global Store Initialization
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `GLOBAL_UNIQUE_ID`    | Unique identifier worker in global store.        | `None`  |
| `GLOBAL_ADDR`         | IP Address of the global store                   | `None`  |
| `GLOBAL_PORT`         | Port number of the global store.                 | `None` |
| `GLOBAL_WORLD_SIZE`   | The size of the global process group.            | `1` |
| `GLOBAL_RANK`         | Rank of the process in the global process group. | `0` |

### Elastic Device Mesh Configuration
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `ZERO_BAND_LOG_LEVEL` | Enable debug mode for loge | `False` |
| `ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS` | Number of seconds before the global store operations timeout | `300` |
| `ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS` | Number of seconds between polls to the store when waiting for values | `0.1` |
| `ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS` | Interval in seconds between heartbeats | `2` |
| `ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS` | Time in seconds after which a node is considered dead if no heartbeat is received | `10` |
| `ZERO_BAND_LIVE_RECO_PORT` | Port number for the live recovery server | random |  
| `ZERO_BAND_LIVE_RECO_ADDR` | IP Address for the live recovery server | `localhost` |  

## Troubleshooting

If you encounter any dataset loading errors at the beginning of training, try setting:

```bash
export HF_HUB_ETAG_TIMEOUT=500
```
