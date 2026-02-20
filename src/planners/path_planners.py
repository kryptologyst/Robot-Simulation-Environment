"""Advanced path planning algorithms for robot navigation."""

from __future__ import annotations

import heapq
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..simulation.environment import RobotSimulationEnvironment, RobotState

logger = logging.getLogger(__name__)


@dataclass
class PathPlanningConfig:
    """Configuration for path planning algorithms."""
    
    # A* specific
    heuristic_weight: float = 1.0
    allow_diagonal: bool = True
    
    # RRT* specific
    max_iterations: int = 1000
    step_size: float = 1.0
    goal_tolerance: float = 1.0
    rewire_radius: float = 2.0
    
    # MPC specific
    horizon_length: int = 10
    control_horizon: int = 5
    max_velocity: float = 2.0
    max_angular_velocity: float = 1.0
    
    # General
    collision_buffer: float = 0.5
    seed: Optional[int] = None


@dataclass
class PathPoint:
    """Represents a point in a planned path."""
    
    position: Tuple[float, float]
    orientation: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    angular_velocity: float = 0.0
    timestamp: float = 0.0
    
    def distance_to(self, other: PathPoint) -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.position[0] - other.position[0]) ** 2 + 
            (self.position[1] - other.position[1]) ** 2
        )


class PathPlanner(ABC):
    """Abstract base class for path planning algorithms."""
    
    def __init__(self, config: PathPlanningConfig) -> None:
        """Initialize the path planner."""
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
    
    @abstractmethod
    def plan(
        self, 
        environment: RobotSimulationEnvironment,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[PathPoint]:
        """
        Plan a path from start to goal.
        
        Args:
            environment: Simulation environment
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of path points
        """
        pass
    
    def smooth_path(self, path: List[PathPoint]) -> List[PathPoint]:
        """
        Smooth a path by removing unnecessary waypoints.
        
        Args:
            path: Original path
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Check if we can skip this point
            prev_point = smoothed[-1]
            next_point = path[i + 1]
            
            # Simple smoothing: skip if angle change is small
            if i < len(path) - 1:
                angle_change = abs(path[i].orientation - prev_point.orientation)
                if angle_change < 0.1:  # Small angle change threshold
                    continue
            
            smoothed.append(path[i])
        
        smoothed.append(path[-1])
        return smoothed


class AStarPlanner(PathPlanner):
    """
    A* path planning algorithm implementation.
    
    A* is an optimal pathfinding algorithm that uses heuristics to efficiently
    find the shortest path from start to goal while avoiding obstacles.
    """
    
    def __init__(self, config: Optional[PathPlanningConfig] = None) -> None:
        """Initialize A* planner."""
        super().__init__(config or PathPlanningConfig())
    
    def plan(
        self, 
        environment: RobotSimulationEnvironment,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[PathPoint]:
        """
        Plan path using A* algorithm.
        
        Args:
            environment: Simulation environment
            start: Starting position
            goal: Goal position
            
        Returns:
            List of path points
        """
        logger.info(f"A* planning from {start} to {goal}")
        
        # Convert to grid coordinates
        start_grid = (int(start[0]), int(start[1]))
        goal_grid = (int(goal[0]), int(goal[1]))
        
        # Priority queue: (f_cost, g_cost, position, parent)
        open_set = [(0, 0, start_grid, None)]
        closed_set: Set[Tuple[int, int]] = set()
        
        # Cost tracking
        g_costs: Dict[Tuple[int, int], float] = {start_grid: 0}
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_grid: None}
        
        # Directions (8-connected if diagonal allowed)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]
        if self.config.allow_diagonal:
            directions.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)])
        
        while open_set:
            f_cost, g_cost, current, parent = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if goal reached
            if self._is_goal_reached(current, goal_grid):
                logger.info("A* goal reached")
                return self._reconstruct_path(parents, current, start_grid, goal)
            
            # Explore neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                # Check bounds and collision
                if not self._is_valid_position(neighbor, environment):
                    continue
                
                # Calculate costs
                move_cost = math.sqrt(dx**2 + dy**2)
                tentative_g = g_cost + move_cost
                
                if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g
                    h_cost = self._heuristic(neighbor, goal_grid)
                    f_cost = tentative_g + self.config.heuristic_weight * h_cost
                    
                    parents[neighbor] = current
                    heapq.heappush(open_set, (f_cost, tentative_g, neighbor, current))
        
        logger.warning("A* failed to find path")
        return []
    
    def _is_valid_position(
        self, 
        position: Tuple[int, int], 
        environment: RobotSimulationEnvironment
    ) -> bool:
        """Check if position is valid (within bounds and no collision)."""
        x, y = position
        if (x < 0 or x >= environment.grid_size[0] or 
            y < 0 or y >= environment.grid_size[1]):
            return False
        
        return not environment.is_collision((float(x), float(y)))
    
    def _is_goal_reached(
        self, 
        current: Tuple[int, int], 
        goal: Tuple[int, int]
    ) -> bool:
        """Check if current position is close enough to goal."""
        return abs(current[0] - goal[0]) <= 1 and abs(current[1] - goal[1]) <= 1
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate heuristic distance to goal."""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def _reconstruct_path(
        self, 
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        current: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[PathPoint]:
        """Reconstruct path from parent pointers."""
        path_points = []
        
        # Reconstruct from goal to start
        while current is not None:
            path_points.append(PathPoint(position=(float(current[0]), float(current[1]))))
            current = parents.get(current)
        
        # Reverse to get start to goal
        path_points.reverse()
        
        # Add goal point if not already included
        if not path_points or path_points[-1].distance_to(PathPoint(position=goal)) > 0.1:
            path_points.append(PathPoint(position=goal))
        
        return path_points


class RRTStarPlanner(PathPlanner):
    """
    RRT* (Rapidly-exploring Random Tree Star) path planning algorithm.
    
    RRT* is a sampling-based algorithm that builds a tree of feasible paths
    and continuously improves the solution quality.
    """
    
    def __init__(self, config: Optional[PathPlanningConfig] = None) -> None:
        """Initialize RRT* planner."""
        super().__init__(config or PathPlanningConfig())
    
    def plan(
        self, 
        environment: RobotSimulationEnvironment,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[PathPoint]:
        """
        Plan path using RRT* algorithm.
        
        Args:
            environment: Simulation environment
            start: Starting position
            goal: Goal position
            
        Returns:
            List of path points
        """
        logger.info(f"RRT* planning from {start} to {goal}")
        
        # Tree structure: {node: parent}
        tree: Dict[Tuple[float, float], Optional[Tuple[float, float]]] = {start: None}
        costs: Dict[Tuple[float, float], float] = {start: 0.0}
        
        goal_reached = False
        
        for iteration in range(self.config.max_iterations):
            # Sample random point
            if random.random() < 0.1:  # 10% chance to sample goal
                random_point = goal
            else:
                random_point = self._sample_random_point(environment)
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_point, tree)
            
            # Extend tree towards random point
            new_node = self._extend_towards(nearest_node, random_point)
            
            if new_node is None:
                continue
            
            # Check if new node is valid
            if environment.is_collision(new_node):
                continue
            
            # Find nearby nodes for rewiring
            nearby_nodes = self._find_nearby_nodes(new_node, tree, costs)
            
            # Choose best parent
            best_parent = nearest_node
            best_cost = costs[nearest_node] + self._distance(nearest_node, new_node)
            
            for nearby_node in nearby_nodes:
                if nearby_node == nearest_node:
                    continue
                
                cost = costs[nearby_node] + self._distance(nearby_node, new_node)
                if cost < best_cost and self._is_collision_free(nearby_node, new_node, environment):
                    best_parent = nearby_node
                    best_cost = cost
            
            # Add node to tree
            tree[new_node] = best_parent
            costs[new_node] = best_cost
            
            # Rewire nearby nodes
            for nearby_node in nearby_nodes:
                if nearby_node == best_parent:
                    continue
                
                new_cost = costs[new_node] + self._distance(new_node, nearby_node)
                if (new_cost < costs[nearby_node] and 
                    self._is_collision_free(new_node, nearby_node, environment)):
                    
                    tree[nearby_node] = new_node
                    costs[nearby_node] = new_cost
            
            # Check if goal reached
            if self._distance(new_node, goal) < self.config.goal_tolerance:
                goal_reached = True
                logger.info(f"RRT* goal reached after {iteration + 1} iterations")
                break
        
        if not goal_reached:
            logger.warning("RRT* failed to reach goal")
            return []
        
        # Reconstruct path
        return self._reconstruct_path(tree, new_node, start)
    
    def _sample_random_point(self, environment: RobotSimulationEnvironment) -> Tuple[float, float]:
        """Sample a random point in the environment."""
        x = random.uniform(0, environment.grid_size[0])
        y = random.uniform(0, environment.grid_size[1])
        return (x, y)
    
    def _find_nearest_node(
        self, 
        point: Tuple[float, float], 
        tree: Dict[Tuple[float, float], Optional[Tuple[float, float]]]
    ) -> Tuple[float, float]:
        """Find the nearest node in the tree to the given point."""
        min_distance = float('inf')
        nearest_node = None
        
        for node in tree.keys():
            distance = self._distance(node, point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_towards(
        self, 
        from_node: Tuple[float, float], 
        to_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Extend tree from from_node towards to_point."""
        distance = self._distance(from_node, to_point)
        
        if distance <= self.config.step_size:
            return to_point
        
        # Calculate unit vector
        dx = (to_point[0] - from_node[0]) / distance
        dy = (to_point[1] - from_node[1]) / distance
        
        # Step towards target
        new_x = from_node[0] + dx * self.config.step_size
        new_y = from_node[1] + dy * self.config.step_size
        
        return (new_x, new_y)
    
    def _find_nearby_nodes(
        self, 
        node: Tuple[float, float], 
        tree: Dict[Tuple[float, float], Optional[Tuple[float, float]]],
        costs: Dict[Tuple[float, float], float]
    ) -> List[Tuple[float, float]]:
        """Find nodes within rewiring radius."""
        nearby_nodes = []
        
        for other_node in tree.keys():
            if self._distance(node, other_node) <= self.config.rewire_radius:
                nearby_nodes.append(other_node)
        
        return nearby_nodes
    
    def _is_collision_free(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        environment: RobotSimulationEnvironment
    ) -> bool:
        """Check if path between two points is collision-free."""
        # Simple line-of-sight check
        num_samples = int(self._distance(start, end) / 0.1) + 1
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            if environment.is_collision((x, y)):
                return False
        
        return True
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _reconstruct_path(
        self, 
        tree: Dict[Tuple[float, float], Optional[Tuple[float, float]]],
        goal_node: Tuple[float, float],
        start: Tuple[float, float]
    ) -> List[PathPoint]:
        """Reconstruct path from tree."""
        path_points = []
        current = goal_node
        
        while current is not None:
            path_points.append(PathPoint(position=current))
            current = tree.get(current)
        
        path_points.reverse()
        return path_points


class MPCPlanner(PathPlanner):
    """
    Model Predictive Control (MPC) path planning algorithm.
    
    MPC solves an optimization problem at each time step to find the optimal
    control sequence while respecting constraints.
    """
    
    def __init__(self, config: Optional[PathPlanningConfig] = None) -> None:
        """Initialize MPC planner."""
        super().__init__(config or PathPlanningConfig())
    
    def plan(
        self, 
        environment: RobotSimulationEnvironment,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> List[PathPoint]:
        """
        Plan path using MPC algorithm.
        
        Args:
            environment: Simulation environment
            start: Starting position
            goal: Goal position
            
        Returns:
            List of path points
        """
        logger.info(f"MPC planning from {start} to {goal}")
        
        # Simple MPC implementation using gradient descent
        path_points = []
        current_pos = np.array(start)
        goal_pos = np.array(goal)
        
        dt = 0.1
        max_steps = 1000
        
        for step in range(max_steps):
            # Check if goal reached
            if np.linalg.norm(current_pos - goal_pos) < self.config.goal_tolerance:
                logger.info(f"MPC goal reached after {step} steps")
                break
            
            # Calculate control input (simple gradient descent)
            direction = goal_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Apply velocity constraints
            velocity = direction * min(self.config.max_velocity, 
                                     np.linalg.norm(goal_pos - current_pos))
            
            # Update position
            new_pos = current_pos + velocity * dt
            
            # Check collision
            if environment.is_collision(tuple(new_pos)):
                # Try alternative directions
                for angle_offset in [0.5, -0.5, 1.0, -1.0]:
                    angle = np.arctan2(direction[1], direction[0]) + angle_offset
                    alt_direction = np.array([np.cos(angle), np.sin(angle)])
                    alt_velocity = alt_direction * self.config.max_velocity * 0.5
                    alt_pos = current_pos + alt_velocity * dt
                    
                    if not environment.is_collision(tuple(alt_pos)):
                        new_pos = alt_pos
                        velocity = alt_velocity
                        break
                else:
                    logger.warning("MPC stuck in collision")
                    break
            
            # Add to path
            path_points.append(PathPoint(
                position=tuple(current_pos),
                velocity=tuple(velocity)
            ))
            
            current_pos = new_pos
        
        # Add final goal point
        path_points.append(PathPoint(position=goal))
        
        return path_points


def create_planner(algorithm: str, config: Optional[PathPlanningConfig] = None) -> PathPlanner:
    """
    Factory function to create path planners.
    
    Args:
        algorithm: Algorithm name ('astar', 'rrtstar', 'mpc')
        config: Planner configuration
        
    Returns:
        Path planner instance
    """
    config = config or PathPlanningConfig()
    
    if algorithm.lower() == 'astar':
        return AStarPlanner(config)
    elif algorithm.lower() == 'rrtstar':
        return RRTStarPlanner(config)
    elif algorithm.lower() == 'mpc':
        return MPCPlanner(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main() -> None:
    """Example usage of path planners."""
    from ..simulation.environment import RobotSimulationEnvironment, SimulationConfig
    
    # Create environment
    config = SimulationConfig(grid_size=(20, 20), seed=42)
    env = RobotSimulationEnvironment(
        config=config,
        start_position=(2, 2),
        goal_position=(18, 18),
        num_obstacles=10
    )
    
    # Test different planners
    planners = ['astar', 'rrtstar', 'mpc']
    
    for planner_name in planners:
        logger.info(f"Testing {planner_name.upper()} planner")
        
        planner = create_planner(planner_name)
        path = planner.plan(env, (2, 2), (18, 18))
        
        logger.info(f"{planner_name.upper()} found path with {len(path)} points")
        
        if path:
            # Visualize path
            env.visualize()


if __name__ == "__main__":
    main()
