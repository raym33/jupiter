# Mac Cluster Setup

This guide explains how to set up a cluster of Apple Silicon Macs for distributed training with Jupiter.

## Hardware Requirements

### Minimum
- 2 Macs with Apple Silicon (M1/M2/M3/M4/M5)
- 16GB RAM per Mac
- Thunderbolt or Ethernet connection between Macs

### Recommended
- 4-5 Macs with M4 or newer
- 16-24GB RAM per Mac
- Thunderbolt 4/5 connection for maximum performance
- 1 Mac dedicated as generator (higher RAM preferred)

## Step 1: Prepare Each Mac

### 1.1 Install System Dependencies

On **each Mac** in the cluster:

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install MPI
brew install mpich
```

### 1.2 Install Jupiter

```bash
# Clone the repository
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Jupiter with Mac dependencies
pip install -e ".[mac]"
```

### 1.3 Verify MLX

```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

## Step 2: Configure Connectivity

### Option A: Thunderbolt (Recommended)

Thunderbolt offers 40Gb/s (TB4) or 80Gb/s (TB5), much faster than Ethernet.

#### Daisy-Chain Topology

```
MacBook ──TB──► Mac Mini 1 ──TB──► Mac Mini 2 ──TB──► Mac Mini 3 ──TB──► Mac Mini 4
```

1. Connect Macs in a chain with Thunderbolt cables
2. macOS will automatically detect the connections
3. Verify in System Preferences > Network

#### Configure Thunderbolt Bridge

For each pair of connected Macs:

1. Open **System Preferences > Network**
2. Select the **Thunderbolt Bridge** connection
3. Configure IP manually:
   - Main Mac: `169.254.1.1`
   - Mac Mini 1: `169.254.1.2`
   - Mac Mini 2: `169.254.1.3`
   - etc.

### Option B: Ethernet

If you can't use Thunderbolt:

1. Connect all Macs to the same Ethernet switch
2. Use static IPs in the same subnet:
   - `192.168.1.10`, `192.168.1.11`, etc.

## Step 3: Configure SSH

For MPI to launch processes on remote nodes, you need passwordless SSH.

### On the main Mac (leader node):

```bash
# Generate SSH key if it doesn't exist
ssh-keygen -t ed25519

# Copy key to each node
ssh-copy-id user@mac-mini-1.local
ssh-copy-id user@mac-mini-2.local
ssh-copy-id user@mac-mini-3.local
ssh-copy-id user@mac-mini-4.local
```

### Verify connection:

```bash
# Should connect without asking for password
ssh mac-mini-1.local "echo 'Connection OK'"
```

## Step 4: Configure the Cluster in Jupiter

Edit `config/cluster.yaml`:

```yaml
backend: "mpi"
use_thunderbolt_bridge: true

nodes:
  - host: "macbook.local"
    name: "Main MacBook"
    device_type: "mac"
    memory_gb: 24
    chip: "M4"
    role: "generator"
    connection_type: "thunderbolt"

  - host: "mac-mini-1.local"
    name: "Mac Mini 1"
    device_type: "mac"
    memory_gb: 16
    chip: "M4"
    role: "trainer"
    connection_type: "thunderbolt"

  # ... more nodes
```

## Step 5: Verify the Cluster

```bash
# On the main Mac
cd jupiter
source venv/bin/activate

# Verify configuration
jupiter check
```

You should see something like:

```
Configuration Files
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ File             ┃ Status                              ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ cluster.yaml     │ ✓ Found                             │
│ Domains          │ ✓ 1 found: unreal_engine            │
└──────────────────┴─────────────────────────────────────┘

Verifying cluster...
✓ Cluster configuration valid

Cluster Nodes
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Name               ┃ Host                 ┃ Type  ┃ RAM   ┃ Role      ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ Main MacBook       │ macbook.local        │ mac   │ 24GB  │ generator │
│ Mac Mini 1         │ mac-mini-1.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 2         │ mac-mini-2.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 3         │ mac-mini-3.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 4         │ mac-mini-4.local     │ mac   │ 16GB  │ trainer   │
└────────────────────┴──────────────────────┴───────┴───────┴───────────┘

Total memory: 88GB
```

## Step 6: Test Distributed Training

```bash
# Quick test with small model
jupiter start --domain unreal_engine --epochs 1
```

## Troubleshooting

### "Connection refused" on SSH

```bash
# Enable Remote Login on the target Mac
# System Preferences > Sharing > Remote Login
```

### "MLX not found" on remote nodes

Make sure the virtual environment is activated on all nodes:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/jupiter/venv/bin:$PATH"
```

### Slow performance

1. Verify you're using Thunderbolt, not Ethernet
2. Check that no other apps are using the network
3. Consider increasing `sync_every_n_steps` if network is slow

## Next Steps

- [Create a new domain](new_domain.md)
- [Add NVIDIA GPUs](nvidia_setup.md)
- [Understand self-improvement](self_improvement.md)
