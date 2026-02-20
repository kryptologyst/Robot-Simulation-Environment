# Robot Simulation Environment

Research-grade robot simulation environment for path planning and navigation research.

## Project Overview

This project implements a comprehensive robot simulation environment focusing on **Planning & Navigation** with advanced path planning algorithms, sensor simulation, and evaluation frameworks.

### Subtype: Planning & Navigation
- **Path Planning**: A*, RRT*, MPC-based planning
- **Motion Planning**: Kinematic constraints, collision avoidance
- **Obstacle Avoidance**: Dynamic and static obstacle handling
- **Multi-robot**: Swarm coordination and formation control

## Features

- **Advanced Path Planning**: A*, RRT*, MPC with constraints
- **Sensor Simulation**: LiDAR, cameras, IMU with realistic noise models
- **Multiple Robot Types**: Differential drive, omnidirectional, aerial
- **Evaluation Framework**: Comprehensive metrics and leaderboards
- **ROS 2 Integration**: Modern robotics middleware support
- **Interactive Demos**: Streamlit-based visualization tools

## Quick Start

### Prerequisites

- Python 3.10+
- ROS 2 Humble (optional, for ROS integration)
- CUDA-capable GPU (optional, for accelerated simulation)

### Installation

```bash
# Clone and setup
git clone https://github.com/kryptologyst/Robot-Simulation-Environment.git
cd Robot-Simulation-Environment

# Install dependencies
pip install -r requirements.txt

# Run basic demo
python scripts/run_demo.py

# Run interactive visualization
streamlit run demo/app.py
```

### Basic Usage

```python
from src.simulation import RobotSimulationEnvironment
from src.planners import AStarPlanner, RRTStarPlanner
from src.robots import DifferentialDriveRobot

# Create simulation environment
env = RobotSimulationEnvironment(
    grid_size=(50, 50),
    start_position=(5, 5),
    goal_position=(45, 45),
    num_obstacles=100
)

# Create robot and planner
robot = DifferentialDriveRobot(position=(5, 5), orientation=0.0)
planner = AStarPlanner()

# Plan and execute path
path = planner.plan(env, robot.position, env.goal_position)
env.execute_path(robot, path)
```

## Repository Structure

```
├── src/                    # Core source code
│   ├── simulation/        # Simulation environment
│   ├── planners/          # Path planning algorithms
│   ├── robots/           # Robot models and controllers
│   ├── sensors/          # Sensor simulation
│   ├── perception/       # Perception pipelines
│   └── utils/            # Utilities and helpers
├── robots/               # Robot descriptions (URDF, meshes)
├── launch/               # ROS 2 launch files
├── config/               # Configuration files
├── data/                 # Datasets and logs
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── assets/               # Generated artifacts
├── demo/                 # Interactive demos
└── docs/                 # Documentation
```

## Algorithms Implemented

### Path Planning
- **A***: Optimal pathfinding with heuristics
- **RRT***: Sampling-based optimal planning
- **MPC**: Model Predictive Control for dynamic environments
- **DWA**: Dynamic Window Approach for local planning

### Motion Planning
- **Kinematic Constraints**: Non-holonomic robot models
- **Collision Avoidance**: Real-time obstacle avoidance
- **Trajectory Optimization**: Smooth path generation

### Multi-robot Coordination
- **Formation Control**: Maintain robot formations
- **Consensus Algorithms**: Distributed decision making
- **Swarm Navigation**: Coordinated group movement

## Evaluation Metrics

### Path Planning Performance
- **Success Rate**: Percentage of successful pathfinding
- **Path Length**: Ratio to optimal path length
- **Planning Time**: Time to compute path
- **Collision Rate**: Percentage of paths with collisions

### Motion Quality
- **Smoothness**: Jerk and acceleration metrics
- **Efficiency**: Energy consumption
- **Accuracy**: Tracking error vs planned path

### Multi-robot Metrics
- **Formation Error**: Deviation from desired formation
- **Communication Load**: Message passing overhead
- **Task Completion**: Success rate for coordinated tasks

## Configuration

All parameters are configurable via YAML files in `config/`:

```yaml
# config/simulation.yaml
simulation:
  grid_size: [50, 50]
  time_step: 0.1
  max_simulation_time: 60.0

robot:
  type: "differential_drive"
  max_velocity: 2.0
  max_angular_velocity: 1.0

planner:
  algorithm: "astar"
  heuristic_weight: 1.0
  collision_buffer: 0.5
```

## Safety and Limitations

**⚠️ IMPORTANT**: This project is for **RESEARCH AND EDUCATION ONLY**. See [DISCLAIMER.md](DISCLAIMER.md) for safety information.

### Known Limitations
- No real-time performance guarantees
- Limited dynamic obstacle handling
- No emergency stop mechanisms
- Experimental algorithms not validated for safety

### Safety Features
- Velocity and acceleration limits
- Collision detection and avoidance
- Simulation-only operation by default
- Comprehensive logging and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting and tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{robot_simulation_environment,
  title={Robot Simulation Environment: A Framework for Path Planning Research},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Robot-Simulation-Environment}
}
```
# Robot-Simulation-Environment
