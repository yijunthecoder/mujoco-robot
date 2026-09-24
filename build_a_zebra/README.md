# Build-a-Zebra

Behavior Tree decision layer for a two-arm robot assembly task. Commands a simulated dual-arm MuJoCo robot to pick up three DUPLO bricks (legs, body, head) and stack them into a zebra, with recovery when things fail.

Built by **Victor** (decision layer) and **Kang** (simulation + robot control). The two halves communicate over ROS 2 topics — same laptop or across two laptops on the same network.

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Run](#run)
- [Interface spec](#interface-spec)
- [File structure](#file-structure)
- [How it works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## What it does

Given a recipe file (`bom.json`) that says "build a zebra: legs → body → head", the system:

1. Waits for perception to report where each brick is
2. Sends a pick command to the robot
3. Waits for the robot to reply `SUCCEEDED` or `FAILED`
4. On success, sends a place command with the assembly pose (stacked)
5. On failure, applies recovery: retry the pick, re-locate the part, or give up gracefully

The robot side (Kang's) actually moves the arm, checks gripper contact, and reports back honestly.

## Architecture

Two independent programs on the same laptop, communicating over three ROS 2 topics.

```text
┌─────────────────────────────┐         ┌─────────────────────────────┐
│   build_a_zebra (C++)       │         │   mujoco-robots (Python)    │
│   "The Brain"               │         │   "The Hands"               │
│                             │         │                             │
│   ┌─────────────────────┐   │         │   ┌─────────────────────┐   │
│   │ Behavior Tree       │   │         │   │ MuJoCo Simulator    │   │
│   │ zebra_tree.xml      │   │         │   │ stationlite dual-arm│   │
│   └──────────┬──────────┘   │         │   └──────────┬──────────┘   │
│              │              │         │              │              │
│   ┌──────────▼──────────┐   │  pick   │   ┌──────────▼──────────┐   │
│   │ SkillBridge         │───┼─────────┼──►│ zebra_skill_bridge  │   │
│   └──────────┬──────────┘   │  place  │   └──────────┬──────────┘   │
│              │              │         │              │              │
│   ┌──────────▼──────────┐   │ coords  │   ┌──────────▼──────────┐   │
│   │ WorldModel          │◄──┼─────────┼───│ zebra_publisher     │   │
│   └─────────────────────┘   │ status  │   └─────────────────────┘   │
│                             │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
              │                                       │
              └──── /zebra/skill_commands ────────────┘
              └──── /zebra/skill_status ──────────────┘
              └──── /zebra/perception_updates ─────────┘
```

Neither half knows anything about the other's internals. They just exchange messages. See [Interface spec](#interface-spec) for details.

## Prerequisites

- Ubuntu 22.04 (or WSL2 on Windows 11)
- ROS 2 Humble
- MuJoCo (Python package)
- BehaviorTree.CPP v4
- nlohmann/json
- Python 3.10 with `rclpy` (comes with ROS 2)

Install once:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ros-base \
  ros-humble-behaviortree-cpp \
  ros-humble-ament-cmake-gtest \
  ros-humble-rmw-cyclonedds-cpp \
  nlohmann-json3-dev \
  libgtest-dev \
  mesa-utils libgl1-mesa-dri

pip3 install mujoco
```

## Build

```bash
cd ~/mujoco-project
colcon build --packages-select build_a_zebra
source install/setup.bash
```

For unit tests:

```bash
colcon build --packages-select build_a_zebra --cmake-args -DBUILD_TESTING=ON
colcon test  --packages-select build_a_zebra
colcon test-result --verbose
```

## Run

Two terminals. Both use the same ROS 2 environment.

### Environment (both terminals)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

`ROS_DOMAIN_ID` must match between both programs. `rmw_cyclonedds_cpp` is more reliable than the default for local loopback.

### Terminal 1 — Kang's stack

```bash
cd ~/kang-repo
export LIBGL_ALWAYS_SOFTWARE=1    # prevents black rendering in WSL
bash run_zebra.sh
```

Kang's `run_zebra.sh` launches:

- `zebra_publisher.py` — reads block positions, publishes on `/zebra/perception_updates`
- `zebra_skill_bridge.py` — subscribes to `/zebra/skill_commands`, drives the MuJoCo arms, replies on `/zebra/skill_status`

Wait for the MuJoCo viewer to open and the console to print "ready" or similar.

### Terminal 2 — the Behavior Tree

```bash
cd ~/mujoco-project
source install/setup.bash
ros2 run build_a_zebra zebra_bt_node
```

You should see:

```text
══════════════════════════════════════════
  ZEBRA BUILD STARTED
══════════════════════════════════════════
  Animal : zebra
  Parts  :
    legs    (31111p0e)
    body    (31111p0f)
    head    (31111p0g)
══════════════════════════════════════════

[INFO] Ticking at 2.0 Hz...

──── ZEBRA  ·  0.0s  ·  step 0 ────
  PART      STATUS        ATTEMPTS  MEANING
  ────────  ────────────  ────────  ───────
  legs      UNKNOWN       0         not yet located
  body      UNKNOWN       0         not yet located
  head      UNKNOWN       0         not yet located
──────────────────────────────────────
```

Then it starts picking parts. When done:

```text
══════════════════════════════════════════
  BUILD SUMMARY  ·  zebra
══════════════════════════════════════════
  PART      STATUS        ATTEMPTS
  ────────  ────────────  ────────
  legs      PLACED        0
  body      PLACED        0
  head      PLACED        0
  ────────  ────────────  ────────
  3 placed, 0 escalated, 3 total
  elapsed: 30.0 s  (60 ticks at 2.0 Hz)
══════════════════════════════════════════
Zebra fully assembled.
```

### Verify both nodes see each other (optional third terminal)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 node list              # should show both /zebra_bt_node and Kang's node
ros2 topic list | grep zebra
ros2 topic hz /zebra/perception_updates    # should show ~5 Hz
```

## Interface spec

All positions are in the MuJoCo world frame, in metres, referring to the brick's origin (not its centre).

All three topics use `std_msgs/String`. Payloads are JSON or CSV — no custom `.msg` types.

### `/zebra/skill_commands` — Brain → Hands

One message per pick/place request.

```json
{
  "command_id": "zebra-1",
  "skill": "pick",
  "part_id": "31111p0e",
  "target": { "x": 0.15, "y": -0.19, "z": -0.86 }
}
```

| Field | Notes |
|---|---|
| `command_id` | Unique per request. Echo back in the reply. |
| `skill` | `"pick"` or `"place"` |
| `part_id` | `31111p0e` (legs), `31111p0f` (body), or `31111p0g` (head) |
| `target` | World-frame brick-origin position in metres |

### `/zebra/skill_status` — Hands → Brain

One reply per command.

```json
{"command_id": "zebra-1", "status": "SUCCEEDED"}
{"command_id": "zebra-1", "status": "FAILED", "message": "grasp slipped"}
```

Anything other than `SUCCEEDED` is treated as `FAILED`.

### `/zebra/perception_updates` — Hands → Brain

One CSV line per part, published at ~5 Hz.

```text
31111p0e,LOCATED,0.15,-0.19,-0.86
31111p0f,LOST,,,
```

Format: `part_id,status,x,y,z`

Statuses:
- `LOCATED` — position is known and valid
- `LOST` — perception cannot currently see it
- `PICKED` — the part is in the gripper
- `PLACED` — the part is on the assembly

## File structure

```text
build_a_zebra/
├── package.xml                 ROS 2 package manifest
├── CMakeLists.txt              Build configuration
├── README.md                   This file
├── INTERFACE.md                Contract with Kang's side
│
├── include/zebra_bt/
│   ├── world_model.hpp         The memory — where each part is
│   ├── task_model.hpp          The recipe reader
│   ├── recovery_manager.hpp    The retry / re-locate / escalate rules
│   └── disturbance_generator.hpp  Simulated bad perception (disabled)
│
├── src/
│   ├── main.cpp                The foreman — wires everything, runs the loop
│   ├── world_model.cpp
│   ├── task_model.cpp
│   ├── recovery_manager.cpp
│   └── disturbance_generator.cpp
│
├── trees/
│   └── zebra_tree.xml          The Behavior Tree — the logic, in XML
│
├── resources/
│   ├── bom.json                What to build (recipe)
│   └── disturbances.json       Fault-injection rates
│
├── test/
│   ├── task_model_test.cpp
│   ├── recovery_manager_test.cpp
│   └── world_model_test.cpp
│
└── scripts/
    ├── mujoco_test_executor.py   Local MuJoCo stand-in (optional)
    ├── kang_bridge_stub.py       Minimal fake Kang (optional)
    └── test_bridge.py            Coordinate-checking test bridge (optional)
```

## How it works

### The tick loop

The `main.cpp` loop runs at 2 Hz. Every tick it:

1. Reads any pending messages from `/zebra/skill_status` and `/zebra/perception_updates`
2. Asks the Behavior Tree to advance one step (`tree.tickOnce()`)
3. Every 5 s, prints the status board

```cpp
while (rclcpp::ok() && !world_model->allPartsResolved()) {
  executor.spin_all(std::chrono::milliseconds(100));
  tree.tickOnce();
  tick_count++;
  if (tick_count % STATUS_EVERY == 0) print_status_board(tick_count);
  rate.sleep();
}
```

### The Behavior Tree

Declared entirely in `trees/zebra_tree.xml`. Three node types are used:

| Node | Meaning |
|---|---|
| `Sequence` | Run children in order; stop at first `FAILURE` |
| `Fallback` | Run children in order; stop at first `SUCCESS` |
| `Condition` | Returns `SUCCESS` or `FAILURE` based on world state |

The top-level tree:

```xml
<Sequence name="BuildZebraSequence">
  <SubTree ID="AssemblePart" name="AssembleLegs" part="31111p0e"/>

  <Fallback name="BodyIfLegsPlaced">
    <Sequence>
      <IsPartEscalated part="31111p0e"/>
      <MarkEscalated part="31111p0f"/>
    </Sequence>
    <SubTree ID="AssemblePart" name="AssembleBody" part="31111p0f"/>
  </Fallback>

  <Fallback name="HeadIfBodyPlaced">
    <Sequence>
      <IsPartEscalated part="31111p0f"/>
      <MarkEscalated part="31111p0g"/>
    </Sequence>
    <SubTree ID="AssemblePart" name="AssembleHead" part="31111p0g"/>
  </Fallback>
</Sequence>
```

Plain English: "Assemble legs. If legs succeeded, assemble body. If body succeeded, assemble head. If any predecessor escalated, mark the dependent part as skipped so the loop can end."

### Each part's assembly

```xml
<Fallback name="PartFallback">
  <IsPartPlaced part="{part}"/>          <!-- already done? skip -->
  <IsPartEscalated part="{part}"/>       <!-- already gave up? skip -->

  <Sequence name="PartAssemblySequence">
    <Fallback name="EnsureLocated">
      <IsPartLocated part="{part}"/>     <!-- know where it is? -->
      <LocatePart part="{part}"/>        <!-- if not, wait for perception -->
    </Fallback>

    <RecoveryPolicy part="{part}">       <!-- retry/re-locate/escalate wrapper -->
      <PickPart part="{part}"/>
    </RecoveryPolicy>

    <PlacePart part="{part}"/>           <!-- to the assembly pose -->
  </Sequence>
</Fallback>
```

### Recovery

`RecoveryManager::decide(part_id, failure_type)` returns one of three actions:

| Action | When | Effect |
|---|---|---|
| `RETRY` | First 3 pick failures | Re-run the pick immediately |
| `RELOCATE` | First 2 part-loss events | Wait for fresh perception, then retry |
| `ESCALATE` | Beyond either limit | Mark part as "needs human" and move on |

Limits are configurable via the constructor:

```cpp
auto recovery_manager = std::make_shared<RecoveryManager>(
  /*max_pick_retries=*/3,
  /*max_relocate_attempts=*/2);
```

### Perception staleness

`WorldModel` timestamps every incoming perception message. If no update arrives for a part in 2 seconds, the position is demoted to `UNKNOWN` so the tree re-locates instead of picking at a stale coordinate.

Implementation: `world_model.cpp`, `_demote_stale_locked()`.

### Dependency check

If the legs can't be placed, the body has nothing to stand on. The tree's `BodyIfLegsPlaced` fallback checks `IsPartEscalated(legs)` first — if true, it runs `MarkEscalated(body)` and skips the body entirely. Same for head-on-body.

This makes the loop terminate cleanly instead of ticking forever with unreachable parts.

## Testing

```bash
colcon build --packages-select build_a_zebra --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select build_a_zebra
colcon test-result --verbose
```

Expected: ~48 tests passing across `task_model_test`, `recovery_manager_test`, and `world_model_test`.

Test coverage:

| File | Tests |
|---|---|
| `task_model_test.cpp` | BOM parsing, role-order sorting, animal filtering, dependency chain, error handling |
| `recovery_manager_test.cpp` | RETRY/RELOCATE/ESCALATE thresholds, independent counters, per-part isolation, escalation de-duplication, custom thresholds |
| `world_model_test.cpp` | All status transitions, position storage, pick-attempt tracking, `allPartsResolved`, staleness demotion |

### Running without Kang's side

For local testing, three stand-in bridges are provided in `scripts/`:

```bash
# Minimal stub — always succeeds
python3 ~/mujoco-project/build_a_zebra/scripts/kang_bridge_stub.py

# Coordinate checker — succeeds only if BT's target matches truth
python3 ~/mujoco-project/build_a_zebra/scripts/test_bridge.py

# Local MuJoCo executor (welded blocks + contact check)
python3 ~/mujoco-project/build_a_zebra/scripts/mujoco_test_executor.py
```

Each in its own terminal, with the same environment as the BT.

## Troubleshooting

### Both nodes run but don't see each other

Check the environment matches in both terminals:

```bash
echo "DOMAIN=$ROS_DOMAIN_ID  RMW=$RMW_IMPLEMENTATION"
```

Both must print the same values. If either is blank, export it:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### "Package 'build_a_zebra' not found"

You forgot to source the workspace:

```bash
source ~/mujoco-project/install/setup.bash
```

Add it to `~/.bashrc` to avoid retyping.

### MuJoCo viewer opens but renders black

Missing software GL:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

Or permanently:

```bash
echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> ~/.bashrc
```

### BT times out waiting for status

Kang's `zebra_skill_bridge.py` isn't running, or is on a different domain. Check:

```bash
ros2 topic info /zebra/skill_commands -v
```

Expected: `Publisher count: 1`, `Subscription count: 1`.

### Perception never arrives

Check the topic:

```bash
ros2 topic echo /zebra/perception_updates
```

If nothing shows, Kang's `zebra_publisher.py` isn't running or is on a different domain.

## Known limitations

- **Physics is approximate.** Kang's side moves the arm through pre-computed joint poses (no real inverse kinematics at runtime) and welds bricks to the gripper during carry. A real robot would use an IK solver.
- **`PLACED` is trusted.** The tree believes the executor's `SUCCEEDED` reply and doesn't verify via camera that the part actually landed on the assembly. In a real system, you'd confirm post-place.
- **Disturbance injection is disabled.** `disturbances.json` exists but all probabilities are set to zero. The feature was built before live perception was available; with real perception from Kang, injecting fake disturbances would just be overwritten by the next real update.
- **Two-laptop networking.** The system runs on one laptop. Cross-laptop discovery works on the same subnet but is blocked by campus Wi-Fi peer isolation (a network policy, not a code issue).
- **Single-threaded tick loop.** The BT advances at a fixed 2 Hz regardless of how fast the executor replies. A more sophisticated scheduler could tick on events instead of a timer.

## Credits

- Behavior Tree + World Model + Recovery — **Victor**
- MuJoCo simulator + robot control + perception — **Kang**
- Interface contract — joint, see `INTERFACE.md`