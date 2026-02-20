"""Unit tests for robot simulation environment."""

import math
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import modules to test
from src.simulation.environment import RobotSimulationEnvironment, SimulationConfig, Obstacle
from src.planners.path_planners import AStarPlanner, RRTStarPlanner, MPCPlanner, PathPlanningConfig
from src.robots.models import DifferentialDriveRobot, OmnidirectionalRobot, RobotConfig, PIDController


class TestSimulationEnvironment:
    """Test cases for simulation environment."""
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        config = SimulationConfig(grid_size=(10, 10), seed=42)
        env = RobotSimulationEnvironment(
            config=config,
            start_position=(1, 1),
            goal_position=(8, 8),
            num_obstacles=5
        )
        
        assert env.grid_size == (10, 10)
        assert np.array_equal(env.start_position, [1, 1])
        assert np.array_equal(env.goal_position, [8, 8])
        assert len(env.obstacles) <= 5  # May be less due to collision avoidance
    
    def test_obstacle_collision_detection(self):
        """Test obstacle collision detection."""
        config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(config=config, num_obstacles=0)
        
        # Add a test obstacle
        obstacle = Obstacle(position=(5, 5), radius=1.0)
        env.obstacles.append(obstacle)
        
        # Test collision detection
        assert env.is_collision((5, 5))  # Center of obstacle
        assert env.is_collision((5.5, 5.5))  # Inside obstacle
        assert not env.is_collision((7, 7))  # Outside obstacle
        assert not env.is_collision((5, 7))  # Outside obstacle
    
    def test_boundary_collision_detection(self):
        """Test boundary collision detection."""
        config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(config=config, num_obstacles=0)
        
        # Test boundary conditions
        assert env.is_collision((-1, 5))  # Left boundary
        assert env.is_collision((5, -1))  # Bottom boundary
        assert env.is_collision((10, 5))  # Right boundary
        assert env.is_collision((5, 10))  # Top boundary
        assert not env.is_collision((5, 5))  # Inside bounds
    
    def test_goal_reaching(self):
        """Test goal reaching detection."""
        config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(
            config=config,
            start_position=(1, 1),
            goal_position=(5, 5),
            num_obstacles=0
        )
        
        # Test goal reaching
        assert env.is_goal_reached((5, 5), tolerance=1.0)  # Exact goal
        assert env.is_goal_reached((5.5, 5.5), tolerance=1.0)  # Within tolerance
        assert not env.is_goal_reached((7, 7), tolerance=1.0)  # Outside tolerance
    
    def test_robot_state_update(self):
        """Test robot state update."""
        config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(config=config, num_obstacles=0)
        
        # Test valid update
        success = env.update_robot_state((5, 5), 0.0)
        assert success
        assert env.robot_state.position == (5, 5)
        assert env.robot_state.orientation == 0.0
        
        # Test collision update
        success = env.update_robot_state((-1, 5), 0.0)  # Out of bounds
        assert not success
    
    def test_heuristic_distance(self):
        """Test heuristic distance calculation."""
        config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(
            config=config,
            start_position=(0, 0),
            goal_position=(3, 4),
            num_obstacles=0
        )
        
        distance = env.get_heuristic_distance((0, 0))
        expected = math.sqrt(3**2 + 4**2)  # 5.0
        assert abs(distance - expected) < 1e-6


class TestPathPlanners:
    """Test cases for path planning algorithms."""
    
    def test_astar_planner(self):
        """Test A* path planner."""
        config = PathPlanningConfig(heuristic_weight=1.0)
        planner = AStarPlanner(config)
        
        # Create simple environment
        sim_config = SimulationConfig(grid_size=(5, 5))
        env = RobotSimulationEnvironment(
            config=sim_config,
            start_position=(0, 0),
            goal_position=(4, 4),
            num_obstacles=0
        )
        
        path = planner.plan(env, (0, 0), (4, 4))
        assert len(path) > 0
        assert path[0].position == (0, 0)  # Start
        assert path[-1].position == (4, 4)  # Goal
    
    def test_rrtstar_planner(self):
        """Test RRT* path planner."""
        config = PathPlanningConfig(max_iterations=100, step_size=1.0)
        planner = RRTStarPlanner(config)
        
        # Create simple environment
        sim_config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(
            config=sim_config,
            start_position=(1, 1),
            goal_position=(8, 8),
            num_obstacles=0
        )
        
        path = planner.plan(env, (1, 1), (8, 8))
        # RRT* might not always find a path in limited iterations
        if path:
            assert len(path) > 0
            assert path[0].position == (1, 1)  # Start
    
    def test_mpc_planner(self):
        """Test MPC path planner."""
        config = PathPlanningConfig(horizon_length=5)
        planner = MPCPlanner(config)
        
        # Create simple environment
        sim_config = SimulationConfig(grid_size=(10, 10))
        env = RobotSimulationEnvironment(
            config=sim_config,
            start_position=(1, 1),
            goal_position=(8, 8),
            num_obstacles=0
        )
        
        path = planner.plan(env, (1, 1), (8, 8))
        assert len(path) > 0
        assert path[0].position == (1, 1)  # Start
        assert path[-1].position == (8, 8)  # Goal


class TestRobotModels:
    """Test cases for robot models."""
    
    def test_differential_drive_robot(self):
        """Test differential drive robot."""
        config = RobotConfig(max_velocity=2.0, max_angular_velocity=1.0)
        robot = DifferentialDriveRobot(config)
        
        # Test initialization
        assert robot.config.max_velocity == 2.0
        assert robot.config.max_angular_velocity == 1.0
        
        # Test reset
        robot.reset(position=(0, 0), orientation=0.0)
        assert robot.state.position == (0, 0)
        assert robot.state.orientation == 0.0
    
    def test_omnidirectional_robot(self):
        """Test omnidirectional robot."""
        config = RobotConfig(max_velocity=2.0)
        robot = OmnidirectionalRobot(config)
        
        # Test initialization
        assert robot.config.max_velocity == 2.0
        
        # Test reset
        robot.reset(position=(1, 1), orientation=0.5)
        assert robot.state.position == (1, 1)
        assert robot.state.orientation == 0.5
    
    def test_robot_dynamics(self):
        """Test robot dynamics update."""
        robot = DifferentialDriveRobot()
        robot.reset(position=(0, 0), orientation=0.0)
        
        # Test forward motion
        from src.robots.models import ControlInput
        control = ControlInput(linear_velocity=1.0, angular_velocity=0.0)
        
        new_state = robot.update_dynamics(control, dt=0.1)
        
        # Should move forward
        assert new_state.position[0] > 0  # Moved in x direction
        assert abs(new_state.position[1]) < 1e-6  # No y movement
        assert new_state.orientation == 0.0  # No rotation
    
    def test_velocity_limits(self):
        """Test velocity limit enforcement."""
        robot = DifferentialDriveRobot()
        
        from src.robots.models import ControlInput
        # Test exceeding velocity limits
        control = ControlInput(linear_velocity=10.0, angular_velocity=5.0)
        
        robot.apply_control(control, dt=0.1)
        
        # Should be limited to max values
        assert abs(robot.control_input.linear_velocity) <= robot.config.max_velocity
        assert abs(robot.control_input.angular_velocity) <= robot.config.max_angular_velocity


class TestControllers:
    """Test cases for robot controllers."""
    
    def test_pid_controller(self):
        """Test PID controller."""
        robot = DifferentialDriveRobot()
        controller = PIDController(robot)
        
        # Test control computation
        control = controller.compute_control(target_position=(5, 5))
        
        assert isinstance(control.linear_velocity, float)
        assert isinstance(control.angular_velocity, float)
    
    def test_controller_reset(self):
        """Test controller reset."""
        robot = DifferentialDriveRobot()
        controller = PIDController(robot)
        
        # Compute some control
        controller.compute_control(target_position=(5, 5))
        
        # Reset controller
        controller.reset()
        
        # State should be reset
        assert controller.prev_error_linear == 0.0
        assert controller.prev_error_angular == 0.0
        assert controller.integral_linear == 0.0
        assert controller.integral_angular == 0.0


class TestIntegration:
    """Integration tests."""
    
    def test_full_simulation(self):
        """Test complete simulation pipeline."""
        # Create environment
        config = SimulationConfig(grid_size=(10, 10), seed=42)
        env = RobotSimulationEnvironment(
            config=config,
            start_position=(1, 1),
            goal_position=(8, 8),
            num_obstacles=5
        )
        
        # Create robot and controller
        robot = create_robot('differential_drive')
        controller = create_controller('pid', robot)
        
        # Create planner
        planner = create_planner('astar')
        
        # Plan path
        path = planner.plan(env, (1, 1), (8, 8))
        
        if path:  # Path might not be found in all cases
            # Execute path
            robot.reset(position=(1, 1), orientation=0.0)
            
            for path_point in path[:5]:  # Execute first 5 points
                control_input = controller.compute_control(path_point.position)
                robot.apply_control(control_input, dt=0.1)
            
            # Should have moved
            assert robot.state.position != (1, 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        from src.utils.evaluation import MetricsCalculator
        
        # Create mock trajectory
        from src.simulation.environment import RobotState
        trajectory = [
            RobotState(position=(0, 0), orientation=0.0, timestamp=0.0),
            RobotState(position=(1, 0), orientation=0.0, timestamp=0.1),
            RobotState(position=(2, 0), orientation=0.0, timestamp=0.2),
        ]
        
        calculator = MetricsCalculator()
        
        # Test path metrics
        path_metrics = calculator.calculate_path_metrics(
            trajectory, (0, 0), (2, 0)
        )
        
        assert path_metrics['success_rate'] == 1.0  # Goal reached
        assert path_metrics['path_length'] == 2.0  # Total distance
        assert path_metrics['path_efficiency'] == 1.0  # Optimal path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
