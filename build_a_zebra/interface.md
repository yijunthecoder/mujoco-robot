# Victor ↔ Kang — ROS 2 Interface Contract

**Purpose:** Freeze the topic names, message schemas, and part-ID mapping
between Victor's Behavior Tree layer (`build_a_zebra`) and Kang's MuJoCo
simulator (`mujoco-robots`).

**Version:** 1.0 — [date]

---

## 1. Network setup (both laptops)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42             # same on both machines
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp