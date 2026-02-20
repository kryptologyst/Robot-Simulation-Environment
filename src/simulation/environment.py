"""Core simulation environment for robot path planning and navigation."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the simulation environment."""
    
    grid_size: Tuple[int, int] = (50, 50)
    time_step: float = 0.1
    max_simulation_time: float = 60.0
    collision_buffer: float = 0.5
    seed: Optional[int] = None
    
    # Safety limits
    max_velocity: float = 2.0
    max_angular_velocity: float = 1.0
    max_acceleration: float = 1.0
    
    # Visualization
    visualize: bool = True
    save_trajectory: bool = True
    
    def __post_init__(self) -> None:
        """Set random seed if provided."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""
    
    position: Tuple[float, float]
    radius: float
    obstacle_type: str = "static"
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the obstacle."""
        distance = np.sqrt(
            (point[0] - self.position[0]) ** 2 + 
            (point[1] - self.position[1]) ** 2
        )
        return distance <= self.radius


@dataclass
class RobotState:
    """Represents the state of a robot."""
    
    position: Tuple[float, float]
    orientation: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    angular_velocity: float = 0.0
    timestamp: float = 0.0
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert state to numpy array."""
        return np.array([
            self.position[0], self.position[1], self.orientation,
            self.velocity[0], self.velocity[1], self.angular_velocity
        ])


class RobotSimulationEnvironment:
    """
    Modern robot simulation environment for path planning and navigation.
    
    This class provides a comprehensive simulation environment with:
    - Grid-based and continuous space support
    - Multiple obstacle types (static/dynamic)
    - Sensor simulation capabilities
    - Safety constraints and limits
    - Comprehensive logging and evaluation
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        start_position: Optional[Tuple[float, float]] = None,
        goal_position: Optional[Tuple[float, float]] = None,
        num_obstacles: int = 20,
    ) -> None:
        """
        Initialize the robot simulation environment.
        
        Args:
            config: Simulation configuration
            start_position: Initial robot position (x, y)
            goal_position: Target goal position (x, y)
            num_obstacles: Number of obstacles to generate
        """
        self.config = config or SimulationConfig()
        
        # Set positions
        self.start_position = np.array(start_position or (5.0, 5.0))
        self.goal_position = np.array(goal_position or (45.0, 45.0))
        
        # Initialize environment
        self.grid_size = self.config.grid_size
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.obstacles: List[Obstacle] = []
        self.robot_state = RobotState(
            position=tuple(self.start_position),
            orientation=0.0
        )
        
        # Simulation state
        self.current_time = 0.0
        self.trajectory: List[RobotState] = []
        self.collision_count = 0
        self.goal_reached = False
        
        # Generate obstacles
        self._generate_obstacles(num_obstacles)
        
        # Log initialization
        logger.info(f"Initialized simulation environment: {self.grid_size}")
        logger.info(f"Start: {self.start_position}, Goal: {self.goal_position}")
        logger.info(f"Obstacles: {len(self.obstacles)}")
    
    def _generate_obstacles(self, num_obstacles: int) -> None:
        """Generate random obstacles in the environment."""
        max_attempts = 1000
        attempts = 0
        
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Random position
            x = np.random.uniform(0, self.grid_size[0])
            y = np.random.uniform(0, self.grid_size[1])
            position = (x, y)
            
            # Random radius
            radius = np.random.uniform(1.0, 3.0)
            
            # Check if obstacle conflicts with start/goal
            start_dist = np.sqrt(
                (position[0] - self.start_position[0]) ** 2 + 
                (position[1] - self.start_position[1]) ** 2
            )
            goal_dist = np.sqrt(
                (position[0] - self.goal_position[0]) ** 2 + 
                (position[1] - self.goal_position[1]) ** 2
            )
            
            if start_dist > radius + 2.0 and goal_dist > radius + 2.0:
                obstacle = Obstacle(position=position, radius=radius)
                self.obstacles.append(obstacle)
                
                # Mark in grid
                self._mark_obstacle_in_grid(obstacle)
    
    def _mark_obstacle_in_grid(self, obstacle: Obstacle) -> None:
        """Mark obstacle area in the grid."""
        x, y = obstacle.position
        radius = obstacle.radius
        
        # Mark grid cells within obstacle radius
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if dist <= radius:
                    self.grid[i, j] = 1.0
    
    def is_collision(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position collides with obstacles.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if collision detected
        """
        # Check bounds
        if (position[0] < 0 or position[0] >= self.grid_size[0] or
            position[1] < 0 or position[1] >= self.grid_size[1]):
            return True
        
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle.contains(position):
                return True
        
        return False
    
    def is_goal_reached(self, position: Tuple[float, float], tolerance: float = 1.0) -> bool:
        """
        Check if goal is reached.
        
        Args:
            position: Current position
            tolerance: Distance tolerance for goal reaching
            
        Returns:
            True if goal is reached
        """
        distance = np.sqrt(
            (position[0] - self.goal_position[0]) ** 2 + 
            (position[1] - self.goal_position[1]) ** 2
        )
        return distance <= tolerance
    
    def get_heuristic_distance(self, position: Tuple[float, float]) -> float:
        """
        Calculate heuristic distance to goal.
        
        Args:
            position: Current position
            
        Returns:
            Euclidean distance to goal
        """
        return np.sqrt(
            (position[0] - self.goal_position[0]) ** 2 + 
            (position[1] - self.goal_position[1]) ** 2
        )
    
    def get_neighbors(self, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get valid neighboring positions.
        
        Args:
            position: Current position
            
        Returns:
            List of valid neighboring positions
        """
        neighbors = []
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # 4-connected
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # 8-connected
        ]
        
        for dx, dy in directions:
            new_pos = (position[0] + dx, position[1] + dy)
            if not self.is_collision(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def update_robot_state(
        self, 
        new_position: Tuple[float, float],
        new_orientation: float,
        velocity: Optional[Tuple[float, float]] = None,
        angular_velocity: Optional[float] = None
    ) -> bool:
        """
        Update robot state with safety checks.
        
        Args:
            new_position: New robot position
            new_orientation: New robot orientation
            velocity: Robot velocity (optional)
            angular_velocity: Robot angular velocity (optional)
            
        Returns:
            True if update successful, False if collision detected
        """
        # Check for collision
        if self.is_collision(new_position):
            self.collision_count += 1
            logger.warning(f"Collision detected at {new_position}")
            return False
        
        # Update state
        self.robot_state.position = new_position
        self.robot_state.orientation = new_orientation
        self.robot_state.timestamp = self.current_time
        
        if velocity is not None:
            self.robot_state.velocity = velocity
        if angular_velocity is not None:
            self.robot_state.angular_velocity = angular_velocity
        
        # Add to trajectory
        self.trajectory.append(RobotState(
            position=new_position,
            orientation=new_orientation,
            velocity=velocity or (0.0, 0.0),
            angular_velocity=angular_velocity or 0.0,
            timestamp=self.current_time
        ))
        
        # Check goal
        if self.is_goal_reached(new_position):
            self.goal_reached = True
            logger.info(f"Goal reached at time {self.current_time}")
        
        return True
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform one simulation step.
        
        Args:
            dt: Time step (uses config default if None)
            
        Returns:
            Dictionary with simulation state
        """
        dt = dt or self.config.time_step
        self.current_time += dt
        
        return {
            "time": self.current_time,
            "robot_state": self.robot_state,
            "goal_reached": self.goal_reached,
            "collision_count": self.collision_count,
            "trajectory_length": len(self.trajectory)
        }
    
    def reset(self) -> None:
        """Reset the simulation environment."""
        self.robot_state = RobotState(
            position=tuple(self.start_position),
            orientation=0.0
        )
        self.current_time = 0.0
        self.trajectory = []
        self.collision_count = 0
        self.goal_reached = False
        
        logger.info("Simulation environment reset")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get simulation performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trajectory:
            return {}
        
        # Path length
        path_length = 0.0
        for i in range(1, len(self.trajectory)):
            prev_pos = self.trajectory[i-1].position
            curr_pos = self.trajectory[i].position
            path_length += np.sqrt(
                (curr_pos[0] - prev_pos[0]) ** 2 + 
                (curr_pos[1] - prev_pos[1]) ** 2
            )
        
        # Optimal path length
        optimal_length = self.get_heuristic_distance(tuple(self.start_position))
        
        # Efficiency metrics
        path_efficiency = optimal_length / path_length if path_length > 0 else 0.0
        
        return {
            "path_length": path_length,
            "optimal_length": optimal_length,
            "path_efficiency": path_efficiency,
            "collision_count": self.collision_count,
            "simulation_time": self.current_time,
            "goal_reached": float(self.goal_reached),
            "trajectory_points": len(self.trajectory)
        }
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the simulation environment.
        
        Args:
            save_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot obstacles
            for obstacle in self.obstacles:
                circle = patches.Circle(
                    obstacle.position, 
                    obstacle.radius, 
                    color='red', 
                    alpha=0.7
                )
                ax.add_patch(circle)
            
            # Plot start and goal
            ax.scatter(*self.start_position, color='green', s=100, label='Start', zorder=5)
            ax.scatter(*self.goal_position, color='blue', s=100, label='Goal', zorder=5)
            
            # Plot trajectory
            if self.trajectory:
                trajectory_x = [state.position[0] for state in self.trajectory]
                trajectory_y = [state.position[1] for state in self.trajectory]
                ax.plot(trajectory_x, trajectory_y, 'b-', alpha=0.7, linewidth=2, label='Path')
            
            # Plot current robot position
            ax.scatter(*self.robot_state.position, color='orange', s=80, label='Robot', zorder=5)
            
            # Formatting
            ax.set_xlim(0, self.grid_size[0])
            ax.set_ylim(0, self.grid_size[1])
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Robot Simulation Environment')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            if self.config.visualize:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")


def main() -> None:
    """Example usage of the simulation environment."""
    # Create configuration
    config = SimulationConfig(
        grid_size=(30, 30),
        seed=42,
        visualize=True
    )
    
    # Create environment
    env = RobotSimulationEnvironment(
        config=config,
        start_position=(5, 5),
        goal_position=(25, 25),
        num_obstacles=15
    )
    
    # Simple random walk simulation
    logger.info("Starting random walk simulation...")
    
    for step in range(1000):
        # Random movement
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        direction = random.choice(directions)
        
        new_pos = (
            env.robot_state.position[0] + direction[0],
            env.robot_state.position[1] + direction[1]
        )
        
        success = env.update_robot_state(new_pos, env.robot_state.orientation)
        
        if not success:
            continue
        
        env.step()
        
        if env.goal_reached:
            logger.info("Goal reached!")
            break
    
    # Get metrics
    metrics = env.get_metrics()
    logger.info(f"Simulation metrics: {metrics}")
    
    # Visualize
    env.visualize()


if __name__ == "__main__":
    main()
