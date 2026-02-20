"""Comprehensive evaluation framework for robot simulation."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

from ..simulation.environment import RobotSimulationEnvironment, RobotState
from ..planners.path_planners import PathPoint
from ..robots.models import RobotModel

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for robot simulation."""
    
    # Path Planning Metrics
    success_rate: float = 0.0
    path_length: float = 0.0
    optimal_length: float = 0.0
    path_efficiency: float = 0.0
    planning_time: float = 0.0
    
    # Motion Quality Metrics
    smoothness_score: float = 0.0
    jerk_magnitude: float = 0.0
    acceleration_magnitude: float = 0.0
    velocity_consistency: float = 0.0
    
    # Control Performance Metrics
    tracking_error: float = 0.0
    control_effort: float = 0.0
    overshoot: float = 0.0
    settling_time: float = 0.0
    
    # Safety Metrics
    collision_count: int = 0
    collision_rate: float = 0.0
    safety_margin: float = 0.0
    
    # Efficiency Metrics
    energy_consumption: float = 0.0
    computation_time: float = 0.0
    memory_usage: float = 0.0
    
    # Multi-robot Metrics (if applicable)
    formation_error: float = 0.0
    communication_load: float = 0.0
    coordination_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'success_rate': self.success_rate,
            'path_length': self.path_length,
            'optimal_length': self.optimal_length,
            'path_efficiency': self.path_efficiency,
            'planning_time': self.planning_time,
            'smoothness_score': self.smoothness_score,
            'jerk_magnitude': self.jerk_magnitude,
            'acceleration_magnitude': self.acceleration_magnitude,
            'velocity_consistency': self.velocity_consistency,
            'tracking_error': self.tracking_error,
            'control_effort': self.control_effort,
            'overshoot': self.overshoot,
            'settling_time': self.settling_time,
            'collision_count': self.collision_count,
            'collision_rate': self.collision_rate,
            'safety_margin': self.safety_margin,
            'energy_consumption': self.energy_consumption,
            'computation_time': self.computation_time,
            'memory_usage': self.memory_usage,
            'formation_error': self.formation_error,
            'communication_load': self.communication_load,
            'coordination_efficiency': self.coordination_efficiency
        }


@dataclass_json
@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    
    algorithm: str
    robot_type: str
    controller_type: str
    environment_config: Dict[str, Any]
    metrics: EvaluationMetrics
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    @staticmethod
    def calculate_path_metrics(
        trajectory: List[RobotState],
        start_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        optimal_path_length: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate path planning metrics.
        
        Args:
            trajectory: Robot trajectory
            start_position: Start position
            goal_position: Goal position
            optimal_path_length: Optimal path length (if known)
            
        Returns:
            Dictionary of path metrics
        """
        if not trajectory:
            return {
                'success_rate': 0.0,
                'path_length': 0.0,
                'optimal_length': 0.0,
                'path_efficiency': 0.0
            }
        
        # Calculate actual path length
        path_length = 0.0
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1].position
            curr_pos = trajectory[i].position
            path_length += math.sqrt(
                (curr_pos[0] - prev_pos[0])**2 + 
                (curr_pos[1] - prev_pos[1])**2
            )
        
        # Calculate optimal path length
        if optimal_path_length is None:
            optimal_path_length = math.sqrt(
                (goal_position[0] - start_position[0])**2 + 
                (goal_position[1] - start_position[1])**2
            )
        
        # Check if goal was reached
        final_position = trajectory[-1].position
        goal_distance = math.sqrt(
            (final_position[0] - goal_position[0])**2 + 
            (final_position[1] - goal_position[1])**2
        )
        success_rate = 1.0 if goal_distance < 1.0 else 0.0
        
        # Calculate efficiency
        path_efficiency = optimal_path_length / path_length if path_length > 0 else 0.0
        
        return {
            'success_rate': success_rate,
            'path_length': path_length,
            'optimal_length': optimal_path_length,
            'path_efficiency': path_efficiency
        }
    
    @staticmethod
    def calculate_motion_metrics(trajectory: List[RobotState]) -> Dict[str, float]:
        """
        Calculate motion quality metrics.
        
        Args:
            trajectory: Robot trajectory
            
        Returns:
            Dictionary of motion metrics
        """
        if len(trajectory) < 3:
            return {
                'smoothness_score': 0.0,
                'jerk_magnitude': 0.0,
                'acceleration_magnitude': 0.0,
                'velocity_consistency': 0.0
            }
        
        # Calculate velocities and accelerations
        velocities = []
        accelerations = []
        jerks = []
        
        for i in range(len(trajectory)):
            state = trajectory[i]
            vx, vy = state.velocity
            velocity_magnitude = math.sqrt(vx**2 + vy**2)
            velocities.append(velocity_magnitude)
        
        # Calculate accelerations
        for i in range(1, len(trajectory)):
            dt = trajectory[i].timestamp - trajectory[i-1].timestamp
            if dt > 0:
                acc = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(acc)
        
        # Calculate jerks
        for i in range(1, len(accelerations)):
            dt = trajectory[i+1].timestamp - trajectory[i].timestamp
            if dt > 0:
                jerk = (accelerations[i] - accelerations[i-1]) / dt
                jerks.append(jerk)
        
        # Calculate metrics
        jerk_magnitude = np.mean(np.abs(jerks)) if jerks else 0.0
        acceleration_magnitude = np.mean(np.abs(accelerations)) if accelerations else 0.0
        
        # Velocity consistency (inverse of standard deviation)
        velocity_consistency = 1.0 / (np.std(velocities) + 1e-8) if velocities else 0.0
        
        # Smoothness score (inverse of jerk)
        smoothness_score = 1.0 / (jerk_magnitude + 1e-8)
        
        return {
            'smoothness_score': smoothness_score,
            'jerk_magnitude': jerk_magnitude,
            'acceleration_magnitude': acceleration_magnitude,
            'velocity_consistency': velocity_consistency
        }
    
    @staticmethod
    def calculate_control_metrics(
        trajectory: List[RobotState],
        planned_path: List[PathPoint]
    ) -> Dict[str, float]:
        """
        Calculate control performance metrics.
        
        Args:
            trajectory: Actual robot trajectory
            planned_path: Planned path
            
        Returns:
            Dictionary of control metrics
        """
        if not trajectory or not planned_path:
            return {
                'tracking_error': 0.0,
                'control_effort': 0.0,
                'overshoot': 0.0,
                'settling_time': 0.0
            }
        
        # Calculate tracking error
        tracking_errors = []
        for state in trajectory:
            # Find closest point on planned path
            min_distance = float('inf')
            for path_point in planned_path:
                distance = math.sqrt(
                    (state.position[0] - path_point.position[0])**2 + 
                    (state.position[1] - path_point.position[1])**2
                )
                min_distance = min(min_distance, distance)
            tracking_errors.append(min_distance)
        
        tracking_error = np.mean(tracking_errors)
        
        # Calculate control effort (sum of velocity magnitudes)
        control_effort = sum(
            math.sqrt(vx**2 + vy**2) 
            for state in trajectory 
            for vx, vy in [state.velocity]
        )
        
        # Calculate overshoot (simplified)
        max_error = max(tracking_errors) if tracking_errors else 0.0
        overshoot = max_error - tracking_error
        
        # Calculate settling time (time to reach 5% of final error)
        if tracking_errors:
            final_error = tracking_errors[-1]
            threshold = 0.05 * final_error
            
            settling_time = 0.0
            for i, error in enumerate(tracking_errors):
                if error <= threshold:
                    settling_time = trajectory[i].timestamp
                    break
        
        return {
            'tracking_error': tracking_error,
            'control_effort': control_effort,
            'overshoot': overshoot,
            'settling_time': settling_time
        }
    
    @staticmethod
    def calculate_safety_metrics(
        trajectory: List[RobotState],
        environment: RobotSimulationEnvironment
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate safety metrics.
        
        Args:
            trajectory: Robot trajectory
            environment: Simulation environment
            
        Returns:
            Dictionary of safety metrics
        """
        collision_count = 0
        min_safety_margin = float('inf')
        
        for state in trajectory:
            # Check collision
            if environment.is_collision(state.position):
                collision_count += 1
            
            # Calculate minimum distance to obstacles
            min_distance = float('inf')
            for obstacle in environment.obstacles:
                distance = math.sqrt(
                    (state.position[0] - obstacle.position[0])**2 + 
                    (state.position[1] - obstacle.position[1])**2
                ) - obstacle.radius
                min_distance = min(min_distance, distance)
            
            min_safety_margin = min(min_safety_margin, min_distance)
        
        collision_rate = collision_count / len(trajectory) if trajectory else 0.0
        
        return {
            'collision_count': collision_count,
            'collision_rate': collision_rate,
            'safety_margin': max(0.0, min_safety_margin)
        }


class EvaluationFramework:
    """Comprehensive evaluation framework for robot simulation."""
    
    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initialize evaluation framework.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_algorithm(
        self,
        algorithm: str,
        robot_type: str,
        controller_type: str,
        environment: RobotSimulationEnvironment,
        planned_path: List[PathPoint],
        trajectory: List[RobotState],
        planning_time: float,
        execution_time: float
    ) -> ExperimentResult:
        """
        Evaluate a single algorithm run.
        
        Args:
            algorithm: Algorithm name
            robot_type: Robot type
            controller_type: Controller type
            environment: Simulation environment
            planned_path: Planned path
            trajectory: Actual trajectory
            planning_time: Time taken for planning
            execution_time: Time taken for execution
            
        Returns:
            Experiment result
        """
        logger.info(f"Evaluating {algorithm} with {robot_type} robot and {controller_type} controller")
        
        # Calculate metrics
        path_metrics = self.metrics_calculator.calculate_path_metrics(
            trajectory,
            tuple(environment.start_position),
            tuple(environment.goal_position)
        )
        
        motion_metrics = self.metrics_calculator.calculate_motion_metrics(trajectory)
        control_metrics = self.metrics_calculator.calculate_control_metrics(trajectory, planned_path)
        safety_metrics = self.metrics_calculator.calculate_safety_metrics(trajectory, environment)
        
        # Create metrics object
        metrics = EvaluationMetrics(
            success_rate=path_metrics['success_rate'],
            path_length=path_metrics['path_length'],
            optimal_length=path_metrics['optimal_length'],
            path_efficiency=path_metrics['path_efficiency'],
            planning_time=planning_time,
            smoothness_score=motion_metrics['smoothness_score'],
            jerk_magnitude=motion_metrics['jerk_magnitude'],
            acceleration_magnitude=motion_metrics['acceleration_magnitude'],
            velocity_consistency=motion_metrics['velocity_consistency'],
            tracking_error=control_metrics['tracking_error'],
            control_effort=control_metrics['control_effort'],
            overshoot=control_metrics['overshoot'],
            settling_time=control_metrics['settling_time'],
            collision_count=safety_metrics['collision_count'],
            collision_rate=safety_metrics['collision_rate'],
            safety_margin=safety_metrics['safety_margin'],
            computation_time=execution_time
        )
        
        # Create experiment result
        result = ExperimentResult(
            algorithm=algorithm,
            robot_type=robot_type,
            controller_type=controller_type,
            environment_config={
                'grid_size': environment.grid_size,
                'num_obstacles': len(environment.obstacles),
                'start_position': tuple(environment.start_position),
                'goal_position': tuple(environment.goal_position)
            },
            metrics=metrics,
            trajectory=[{
                'x': state.position[0],
                'y': state.position[1],
                'orientation': state.orientation,
                'timestamp': state.timestamp
            } for state in trajectory],
            execution_time=execution_time
        )
        
        self.results.append(result)
        return result
    
    def generate_leaderboard(self) -> pd.DataFrame:
        """
        Generate a leaderboard of algorithm performance.
        
        Returns:
            DataFrame with algorithm rankings
        """
        if not self.results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            row = {
                'Algorithm': result.algorithm,
                'Robot Type': result.robot_type,
                'Controller': result.controller_type,
                **result.metrics.to_dict()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate composite score (weighted combination of key metrics)
        weights = {
            'success_rate': 0.3,
            'path_efficiency': 0.2,
            'smoothness_score': 0.15,
            'tracking_error': -0.1,  # Lower is better
            'collision_rate': -0.15,  # Lower is better
            'planning_time': -0.1  # Lower is better
        }
        
        df['Composite Score'] = 0.0
        for metric, weight in weights.items():
            if metric in df.columns:
                # Normalize metric to [0, 1] range
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min() + 1e-8)
                df['Composite Score'] += weight * normalized
        
        # Sort by composite score
        df = df.sort_values('Composite Score', ascending=False)
        
        return df
    
    def save_results(self, filename: Optional[str] = None) -> None:
        """
        Save evaluation results to file.
        
        Args:
            filename: Output filename
        """
        if not filename:
            filename = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to JSON-serializable format
        results_data = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filename: str) -> None:
        """
        Load evaluation results from file.
        
        Args:
            filename: Input filename
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.results = [ExperimentResult.from_dict(data) for data in results_data]
        logger.info(f"Loaded {len(self.results)} results from {filepath}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Markdown report string
        """
        if not self.results:
            return "# Evaluation Report\n\nNo results available."
        
        leaderboard = self.generate_leaderboard()
        
        report = "# Robot Simulation Evaluation Report\n\n"
        report += f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Experiments:** {len(self.results)}\n\n"
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        report += f"- **Algorithms Tested:** {len(leaderboard['Algorithm'].unique())}\n"
        report += f"- **Robot Types:** {len(leaderboard['Robot Type'].unique())}\n"
        report += f"- **Controllers:** {len(leaderboard['Controller'].unique())}\n\n"
        
        # Top performers
        report += "## Top Performers\n\n"
        report += "| Rank | Algorithm | Robot Type | Controller | Composite Score | Success Rate | Path Efficiency |\n"
        report += "|------|-----------|------------|------------|-----------------|--------------|-----------------|\n"
        
        for i, (_, row) in enumerate(leaderboard.head(10).iterrows()):
            report += f"| {i+1} | {row['Algorithm']} | {row['Robot Type']} | {row['Controller']} | "
            report += f"{row['Composite Score']:.3f} | {row['success_rate']:.3f} | {row['path_efficiency']:.3f} |\n"
        
        report += "\n"
        
        # Detailed metrics
        report += "## Detailed Metrics\n\n"
        metrics = ['success_rate', 'path_efficiency', 'smoothness_score', 'tracking_error', 'collision_rate']
        
        for metric in metrics:
            if metric in leaderboard.columns:
                report += f"### {metric.replace('_', ' ').title()}\n\n"
                report += f"- **Best:** {leaderboard[metric].max():.3f}\n"
                report += f"- **Worst:** {leaderboard[metric].min():.3f}\n"
                report += f"- **Average:** {leaderboard[metric].mean():.3f}\n"
                report += f"- **Std Dev:** {leaderboard[metric].std():.3f}\n\n"
        
        return report


def main() -> None:
    """Example usage of evaluation framework."""
    from ..simulation.environment import RobotSimulationEnvironment, SimulationConfig
    from ..planners.path_planners import create_planner
    from ..robots.models import create_robot, create_controller
    
    # Create evaluation framework
    evaluator = EvaluationFramework()
    
    # Test different algorithms
    algorithms = ['astar', 'rrtstar', 'mpc']
    robot_types = ['differential_drive', 'omnidirectional']
    controllers = ['pid', 'pure_pursuit']
    
    for algorithm in algorithms:
        for robot_type in robot_types:
            for controller_type in controllers:
                logger.info(f"Testing {algorithm} with {robot_type} robot and {controller_type} controller")
                
                # Create environment
                config = SimulationConfig(grid_size=(20, 20), seed=42)
                env = RobotSimulationEnvironment(
                    config=config,
                    start_position=(2, 2),
                    goal_position=(18, 18),
                    num_obstacles=10
                )
                
                # Create robot and controller
                robot = create_robot(robot_type)
                controller = create_controller(controller_type, robot)
                
                # Plan path
                planner = create_planner(algorithm)
                start_time = time.time()
                planned_path = planner.plan(env, (2, 2), (18, 18))
                planning_time = time.time() - start_time
                
                if not planned_path:
                    logger.warning(f"No path found for {algorithm}")
                    continue
                
                # Execute path
                robot.reset(position=(2, 2), orientation=0.0)
                trajectory = []
                execution_start = time.time()
                
                for path_point in planned_path:
                    control_input = controller.compute_control(path_point.position)
                    robot.apply_control(control_input, dt=0.1)
                    trajectory.append(robot.state)
                
                execution_time = time.time() - execution_start
                
                # Evaluate
                result = evaluator.evaluate_algorithm(
                    algorithm, robot_type, controller_type, env,
                    planned_path, trajectory, planning_time, execution_time
                )
                
                logger.info(f"Evaluation complete: Success rate = {result.metrics.success_rate:.3f}")
    
    # Generate report
    leaderboard = evaluator.generate_leaderboard()
    print("\nLeaderboard:")
    print(leaderboard[['Algorithm', 'Robot Type', 'Controller', 'Composite Score', 'success_rate']].head())
    
    # Save results
    evaluator.save_results()
    
    # Generate report
    report = evaluator.generate_report()
    with open("evaluation_report.md", "w") as f:
        f.write(report)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
