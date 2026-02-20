"""Interactive Streamlit demo for robot simulation environment."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.environment import RobotSimulationEnvironment, SimulationConfig
from src.planners.path_planners import create_planner, PathPlanningConfig
from src.robots.models import create_robot, create_controller, RobotConfig
from src.utils.evaluation import EvaluationFramework, MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Robot Simulation Environment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'current_simulation' not in st.session_state:
        st.session_state.current_simulation = None


def create_simulation_environment(config: Dict) -> RobotSimulationEnvironment:
    """Create simulation environment from configuration."""
    sim_config = SimulationConfig(
        grid_size=(config['grid_width'], config['grid_height']),
        seed=config['seed'],
        visualize=False
    )
    
    return RobotSimulationEnvironment(
        config=sim_config,
        start_position=(config['start_x'], config['start_y']),
        goal_position=(config['goal_x'], config['goal_y']),
        num_obstacles=config['num_obstacles']
    )


def run_simulation(
    environment: RobotSimulationEnvironment,
    algorithm: str,
    robot_type: str,
    controller_type: str
) -> Dict:
    """Run a single simulation."""
    # Create robot and controller
    robot_config = RobotConfig(
        max_velocity=2.0,
        max_angular_velocity=1.0
    )
    robot = create_robot(robot_type, robot_config)
    controller = create_controller(controller_type, robot)
    
    # Create planner
    planner_config = PathPlanningConfig(
        heuristic_weight=1.0,
        max_iterations=1000,
        step_size=1.0
    )
    planner = create_planner(algorithm, planner_config)
    
    # Plan path
    start_time = time.time()
    planned_path = planner.plan(
        environment,
        tuple(environment.start_position),
        tuple(environment.goal_position)
    )
    planning_time = time.time() - start_time
    
    if not planned_path:
        return {
            'success': False,
            'error': 'No path found',
            'trajectory': [],
            'planned_path': [],
            'metrics': {}
        }
    
    # Execute path
    robot.reset(
        position=tuple(environment.start_position),
        orientation=0.0
    )
    
    trajectory = []
    execution_start = time.time()
    
    for path_point in planned_path:
        control_input = controller.compute_control(path_point.position)
        robot.apply_control(control_input, dt=0.1)
        trajectory.append(robot.state)
    
    execution_time = time.time() - execution_start
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    path_metrics = metrics_calc.calculate_path_metrics(
        trajectory,
        tuple(environment.start_position),
        tuple(environment.goal_position)
    )
    
    motion_metrics = metrics_calc.calculate_motion_metrics(trajectory)
    safety_metrics = metrics_calc.calculate_safety_metrics(trajectory, environment)
    
    return {
        'success': True,
        'trajectory': trajectory,
        'planned_path': planned_path,
        'metrics': {
            **path_metrics,
            **motion_metrics,
            **safety_metrics,
            'planning_time': planning_time,
            'execution_time': execution_time
        }
    }


def plot_simulation_results(environment: RobotSimulationEnvironment, result: Dict):
    """Plot simulation results."""
    if not result['success']:
        st.error(f"Simulation failed: {result['error']}")
        return
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Environment & Path', 'Performance Metrics'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Plot environment and path
    trajectory = result['trajectory']
    planned_path = result['planned_path']
    
    if trajectory:
        traj_x = [state.position[0] for state in trajectory]
        traj_y = [state.position[1] for state in trajectory]
        
        fig.add_trace(
            go.Scatter(
                x=traj_x, y=traj_y,
                mode='lines+markers',
                name='Actual Path',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
    
    if planned_path:
        plan_x = [point.position[0] for point in planned_path]
        plan_y = [point.position[1] for point in planned_path]
        
        fig.add_trace(
            go.Scatter(
                x=plan_x, y=plan_y,
                mode='lines+markers',
                name='Planned Path',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=3)
            ),
            row=1, col=1
        )
    
    # Plot obstacles
    for obstacle in environment.obstacles:
        fig.add_trace(
            go.Scatter(
                x=[obstacle.position[0]], y=[obstacle.position[1]],
                mode='markers',
                marker=dict(
                    size=obstacle.radius * 10,
                    color='red',
                    opacity=0.7,
                    symbol='circle'
                ),
                name='Obstacle',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot start and goal
    fig.add_trace(
        go.Scatter(
            x=[environment.start_position[0]], y=[environment.start_position[1]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Start',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[environment.goal_position[0]], y=[environment.goal_position[1]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='star'),
            name='Goal',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Plot metrics
    metrics = result['metrics']
    metric_names = ['Success Rate', 'Path Efficiency', 'Smoothness Score', 'Safety Margin']
    metric_values = [
        metrics.get('success_rate', 0),
        metrics.get('path_efficiency', 0),
        metrics.get('smoothness_score', 0),
        metrics.get('safety_margin', 0)
    ]
    
    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=metric_values,
            name='Metrics',
            marker_color=['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in metric_values]
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Simulation Results",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_yaxes(title_text="Y Position", row=1, col=1)
    fig.update_xaxes(title_text="Metrics", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Robot Simulation Environment</h1>', unsafe_allow_html=True)
    
    # Safety warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ SAFETY WARNING:</strong> This simulation is for research and education purposes only. 
        Do not use on real robots without expert review and safety measures.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Simulation Configuration")
    
    # Environment parameters
    st.sidebar.subheader("Environment")
    grid_width = st.sidebar.slider("Grid Width", 10, 50, 30)
    grid_height = st.sidebar.slider("Grid Height", 10, 50, 30)
    num_obstacles = st.sidebar.slider("Number of Obstacles", 5, 30, 15)
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Start and goal positions
    st.sidebar.subheader("Start Position")
    start_x = st.sidebar.slider("Start X", 1, grid_width-2, 2)
    start_y = st.sidebar.slider("Start Y", 1, grid_height-2, 2)
    
    st.sidebar.subheader("Goal Position")
    goal_x = st.sidebar.slider("Goal X", 1, grid_width-2, grid_width-2)
    goal_y = st.sidebar.slider("Goal Y", 1, grid_height-2, grid_height-2)
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm")
    algorithm = st.sidebar.selectbox(
        "Path Planning Algorithm",
        ["astar", "rrtstar", "mpc"],
        help="Choose the path planning algorithm"
    )
    
    # Robot configuration
    st.sidebar.subheader("Robot Configuration")
    robot_type = st.sidebar.selectbox(
        "Robot Type",
        ["differential_drive", "omnidirectional"],
        help="Choose the robot type"
    )
    
    controller_type = st.sidebar.selectbox(
        "Controller Type",
        ["pid", "pure_pursuit"],
        help="Choose the controller type"
    )
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Create environment
            config = {
                'grid_width': grid_width,
                'grid_height': grid_height,
                'num_obstacles': num_obstacles,
                'seed': seed,
                'start_x': start_x,
                'start_y': start_y,
                'goal_x': goal_x,
                'goal_y': goal_y
            }
            
            environment = create_simulation_environment(config)
            
            # Run simulation
            result = run_simulation(environment, algorithm, robot_type, controller_type)
            
            # Store results
            st.session_state.current_simulation = {
                'environment': environment,
                'result': result,
                'config': config,
                'algorithm': algorithm,
                'robot_type': robot_type,
                'controller_type': controller_type
            }
    
    # Main content area
    if st.session_state.current_simulation:
        sim_data = st.session_state.current_simulation
        
        # Display results
        st.header("Simulation Results")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Success Rate",
                f"{sim_data['result']['metrics'].get('success_rate', 0):.2%}"
            )
        
        with col2:
            st.metric(
                "Path Efficiency",
                f"{sim_data['result']['metrics'].get('path_efficiency', 0):.2f}"
            )
        
        with col3:
            st.metric(
                "Planning Time",
                f"{sim_data['result']['metrics'].get('planning_time', 0):.3f}s"
            )
        
        with col4:
            st.metric(
                "Execution Time",
                f"{sim_data['result']['metrics'].get('execution_time', 0):.3f}s"
            )
        
        # Plot results
        plot_simulation_results(sim_data['environment'], sim_data['result'])
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        metrics_df = pd.DataFrame([sim_data['result']['metrics']])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Save results option
        if st.button("💾 Save Results"):
            st.session_state.simulation_results.append(sim_data)
            st.success("Results saved!")
    
    else:
        # Welcome message
        st.info("👈 Configure the simulation parameters in the sidebar and click 'Run Simulation' to start!")
        
        # Algorithm comparison
        st.header("Algorithm Comparison")
        
        st.markdown("""
        ### Available Algorithms
        
        **A*** - Optimal pathfinding algorithm using heuristics
        - Guarantees shortest path
        - Good for static environments
        - Can be slow for large grids
        
        **RRT*** - Sampling-based probabilistic algorithm
        - Works well in complex environments
        - Continuously improves solution quality
        - Good for dynamic environments
        
        **MPC** - Model Predictive Control
        - Handles constraints well
        - Good for real-time control
        - Can handle dynamic obstacles
        """)
        
        # Performance tips
        st.header("Performance Tips")
        
        st.markdown("""
        - **Smaller grids** (10x10 to 20x20) run faster
        - **Fewer obstacles** (5-15) are easier to navigate
        - **A*** works best for simple environments
        - **RRT*** is better for complex obstacle fields
        - **MPC** is good for real-time applications
        """)


if __name__ == "__main__":
    main()
