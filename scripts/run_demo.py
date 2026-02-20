"""Main script to run robot simulation demos."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.simulation.environment import RobotSimulationEnvironment, SimulationConfig
from src.planners.path_planners import create_planner, PathPlanningConfig
from src.robots.models import create_robot, create_controller, RobotConfig
from src.utils.evaluation import EvaluationFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_basic_demo():
    """Run basic simulation demo."""
    logger.info("Running basic simulation demo...")
    
    # Create configuration
    config = SimulationConfig(
        grid_size=(20, 20),
        seed=42,
        visualize=True
    )
    
    # Create environment
    env = RobotSimulationEnvironment(
        config=config,
        start_position=(2, 2),
        goal_position=(18, 18),
        num_obstacles=10
    )
    
    # Create robot and controller
    robot = create_robot('differential_drive')
    controller = create_controller('pid', robot)
    
    # Create planner
    planner = create_planner('astar')
    
    # Plan path
    planned_path = planner.plan(env, (2, 2), (18, 18))
    
    if not planned_path:
        logger.error("No path found!")
        return
    
    logger.info(f"Path planned with {len(planned_path)} points")
    
    # Execute path
    robot.reset(position=(2, 2), orientation=0.0)
    
    for path_point in planned_path:
        control_input = controller.compute_control(path_point.position)
        robot.apply_control(control_input, dt=0.1)
    
    # Get metrics
    metrics = env.get_metrics()
    logger.info(f"Simulation metrics: {metrics}")
    
    # Visualize
    env.visualize()


def run_algorithm_comparison():
    """Run comparison of different algorithms."""
    logger.info("Running algorithm comparison...")
    
    algorithms = ['astar', 'rrtstar', 'mpc']
    evaluator = EvaluationFramework()
    
    for algorithm in algorithms:
        logger.info(f"Testing {algorithm}...")
        
        # Create environment
        config = SimulationConfig(grid_size=(15, 15), seed=42)
        env = RobotSimulationEnvironment(
            config=config,
            start_position=(2, 2),
            goal_position=(13, 13),
            num_obstacles=8
        )
        
        # Create robot and controller
        robot = create_robot('differential_drive')
        controller = create_controller('pid', robot)
        
        # Create planner
        planner = create_planner(algorithm)
        
        # Plan and execute
        planned_path = planner.plan(env, (2, 2), (13, 13))
        
        if planned_path:
            robot.reset(position=(2, 2), orientation=0.0)
            trajectory = []
            
            for path_point in planned_path:
                control_input = controller.compute_control(path_point.position)
                robot.apply_control(control_input, dt=0.1)
                trajectory.append(robot.state)
            
            # Evaluate
            result = evaluator.evaluate_algorithm(
                algorithm, 'differential_drive', 'pid', env,
                planned_path, trajectory, 0.0, 0.0
            )
            
            logger.info(f"{algorithm}: Success rate = {result.metrics.success_rate:.3f}")
    
    # Generate leaderboard
    leaderboard = evaluator.generate_leaderboard()
    print("\nAlgorithm Comparison Results:")
    print(leaderboard[['Algorithm', 'success_rate', 'path_efficiency', 'smoothness_score']].head())


def run_interactive_demo():
    """Run interactive Streamlit demo."""
    logger.info("Starting interactive demo...")
    
    import subprocess
    import os
    
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if demo_path.exists():
        os.chdir(Path(__file__).parent)
        subprocess.run(["streamlit", "run", str(demo_path)])
    else:
        logger.error("Demo app not found!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Simulation Environment")
    parser.add_argument(
        "mode",
        choices=["basic", "comparison", "interactive"],
        help="Demo mode to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.mode == "basic":
            run_basic_demo()
        elif args.mode == "comparison":
            run_algorithm_comparison()
        elif args.mode == "interactive":
            run_interactive_demo()
        
        logger.info("Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
