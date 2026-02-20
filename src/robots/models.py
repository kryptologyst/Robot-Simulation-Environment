"""Robot models and controllers for simulation."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class RobotConfig:
    """Configuration for robot models."""
    
    # Physical parameters
    wheel_base: float = 0.5  # Distance between wheels
    wheel_radius: float = 0.1
    max_velocity: float = 2.0
    max_angular_velocity: float = 1.0
    max_acceleration: float = 1.0
    
    # Control parameters
    control_frequency: float = 10.0  # Hz
    position_tolerance: float = 0.1
    orientation_tolerance: float = 0.1
    
    # Safety limits
    emergency_stop_distance: float = 0.5
    collision_buffer: float = 0.2


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
    
    @classmethod
    def from_array(cls, state_array: npt.NDArray[np.float64]) -> RobotState:
        """Create state from numpy array."""
        return cls(
            position=(state_array[0], state_array[1]),
            orientation=state_array[2],
            velocity=(state_array[3], state_array[4]),
            angular_velocity=state_array[5]
        )


@dataclass
class ControlInput:
    """Represents control input to robot."""
    
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    timestamp: float = 0.0
    
    def to_array(self) -> npt.NDArray[np.float64]:
        """Convert to numpy array."""
        return np.array([self.linear_velocity, self.angular_velocity])


class RobotModel(ABC):
    """Abstract base class for robot models."""
    
    def __init__(self, config: RobotConfig) -> None:
        """Initialize robot model."""
        self.config = config
        self.state = RobotState(position=(0.0, 0.0), orientation=0.0)
        self.control_input = ControlInput()
    
    @abstractmethod
    def update_dynamics(self, control_input: ControlInput, dt: float) -> RobotState:
        """
        Update robot dynamics based on control input.
        
        Args:
            control_input: Control commands
            dt: Time step
            
        Returns:
            Updated robot state
        """
        pass
    
    @abstractmethod
    def get_kinematics(self) -> npt.NDArray[np.float64]:
        """
        Get robot kinematics matrix.
        
        Returns:
            Kinematics matrix
        """
        pass
    
    def apply_control(self, control_input: ControlInput, dt: float) -> RobotState:
        """
        Apply control input with safety checks.
        
        Args:
            control_input: Control commands
            dt: Time step
            
        Returns:
            Updated robot state
        """
        # Apply velocity limits
        limited_input = ControlInput(
            linear_velocity=max(-self.config.max_velocity, 
                              min(self.config.max_velocity, control_input.linear_velocity)),
            angular_velocity=max(-self.config.max_angular_velocity,
                               min(self.config.max_angular_velocity, control_input.angular_velocity)),
            timestamp=control_input.timestamp
        )
        
        self.control_input = limited_input
        self.state = self.update_dynamics(limited_input, dt)
        
        return self.state
    
    def reset(self, position: Tuple[float, float], orientation: float = 0.0) -> None:
        """Reset robot to initial state."""
        self.state = RobotState(position=position, orientation=orientation)
        self.control_input = ControlInput()


class DifferentialDriveRobot(RobotModel):
    """
    Differential drive robot model.
    
    This robot has two wheels that can be controlled independently,
    allowing for forward/backward motion and rotation.
    """
    
    def __init__(self, config: Optional[RobotConfig] = None) -> None:
        """Initialize differential drive robot."""
        super().__init__(config or RobotConfig())
    
    def update_dynamics(self, control_input: ControlInput, dt: float) -> RobotState:
        """
        Update differential drive robot dynamics.
        
        Args:
            control_input: Control commands (linear_velocity, angular_velocity)
            dt: Time step
            
        Returns:
            Updated robot state
        """
        v = control_input.linear_velocity
        omega = control_input.angular_velocity
        
        # Current state
        x, y = self.state.position
        theta = self.state.orientation
        
        # Update position and orientation
        new_x = x + v * math.cos(theta) * dt
        new_y = y + v * math.sin(theta) * dt
        new_theta = theta + omega * dt
        
        # Normalize orientation
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        
        # Update velocities
        new_vx = v * math.cos(new_theta)
        new_vy = v * math.sin(new_theta)
        
        return RobotState(
            position=(new_x, new_y),
            orientation=new_theta,
            velocity=(new_vx, new_vy),
            angular_velocity=omega,
            timestamp=self.state.timestamp + dt
        )
    
    def get_kinematics(self) -> npt.NDArray[np.float64]:
        """
        Get differential drive kinematics matrix.
        
        Returns:
            3x2 kinematics matrix
        """
        theta = self.state.orientation
        
        return np.array([
            [math.cos(theta), 0],
            [math.sin(theta), 0],
            [0, 1]
        ])
    
    def wheel_velocities(self, control_input: ControlInput) -> Tuple[float, float]:
        """
        Convert control input to wheel velocities.
        
        Args:
            control_input: Control commands
            
        Returns:
            Tuple of (left_wheel_velocity, right_wheel_velocity)
        """
        v = control_input.linear_velocity
        omega = control_input.angular_velocity
        
        # Convert to wheel velocities
        left_velocity = v - omega * self.config.wheel_base / 2
        right_velocity = v + omega * self.config.wheel_base / 2
        
        return left_velocity, right_velocity


class OmnidirectionalRobot(RobotModel):
    """
    Omnidirectional robot model.
    
    This robot can move in any direction without changing orientation,
    using mecanum wheels or similar omnidirectional drive system.
    """
    
    def __init__(self, config: Optional[RobotConfig] = None) -> None:
        """Initialize omnidirectional robot."""
        super().__init__(config or RobotConfig())
    
    def update_dynamics(self, control_input: ControlInput, dt: float) -> RobotState:
        """
        Update omnidirectional robot dynamics.
        
        Args:
            control_input: Control commands (vx, vy, omega)
            dt: Time step
            
        Returns:
            Updated robot state
        """
        # For omnidirectional robot, we interpret control input differently
        vx = control_input.linear_velocity  # Forward velocity
        vy = control_input.angular_velocity  # Lateral velocity (reusing angular_velocity field)
        
        # Current state
        x, y = self.state.position
        theta = self.state.orientation
        
        # Update position (omnidirectional movement)
        new_x = x + vx * dt
        new_y = y + vy * dt
        
        # Orientation can be controlled independently
        omega = 0.0  # Could be added as separate control input
        new_theta = theta + omega * dt
        
        return RobotState(
            position=(new_x, new_y),
            orientation=new_theta,
            velocity=(vx, vy),
            angular_velocity=omega,
            timestamp=self.state.timestamp + dt
        )
    
    def get_kinematics(self) -> npt.NDArray[np.float64]:
        """
        Get omnidirectional robot kinematics matrix.
        
        Returns:
            3x2 kinematics matrix
        """
        return np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ])


class RobotController(ABC):
    """Abstract base class for robot controllers."""
    
    def __init__(self, robot: RobotModel) -> None:
        """Initialize controller."""
        self.robot = robot
    
    @abstractmethod
    def compute_control(
        self, 
        target_position: Tuple[float, float],
        target_orientation: Optional[float] = None
    ) -> ControlInput:
        """
        Compute control input to reach target.
        
        Args:
            target_position: Target position (x, y)
            target_orientation: Target orientation (optional)
            
        Returns:
            Control input
        """
        pass


class PIDController(RobotController):
    """
    PID controller for robot navigation.
    
    Uses proportional-integral-derivative control to navigate
    the robot to target positions.
    """
    
    def __init__(
        self, 
        robot: RobotModel,
        kp_linear: float = 1.0,
        ki_linear: float = 0.0,
        kd_linear: float = 0.1,
        kp_angular: float = 2.0,
        ki_angular: float = 0.0,
        kd_angular: float = 0.1
    ) -> None:
        """Initialize PID controller."""
        super().__init__(robot)
        
        # PID gains
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular
        
        # PID state
        self.prev_error_linear = 0.0
        self.prev_error_angular = 0.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.prev_time = 0.0
    
    def compute_control(
        self, 
        target_position: Tuple[float, float],
        target_orientation: Optional[float] = None
    ) -> ControlInput:
        """
        Compute PID control input.
        
        Args:
            target_position: Target position
            target_orientation: Target orientation
            
        Returns:
            Control input
        """
        current_time = self.robot.state.timestamp
        dt = current_time - self.prev_time if self.prev_time > 0 else 0.1
        
        # Position error
        current_pos = self.robot.state.position
        error_x = target_position[0] - current_pos[0]
        error_y = target_position[1] - current_pos[1]
        
        # Distance to target
        distance = math.sqrt(error_x**2 + error_y**2)
        
        # Linear velocity control
        error_linear = distance
        self.integral_linear += error_linear * dt
        derivative_linear = (error_linear - self.prev_error_linear) / dt
        
        linear_velocity = (
            self.kp_linear * error_linear +
            self.ki_linear * self.integral_linear +
            self.kd_linear * derivative_linear
        )
        
        # Angular velocity control
        if target_orientation is not None:
            error_angular = self._normalize_angle(target_orientation - self.robot.state.orientation)
        else:
            # Point towards target
            target_angle = math.atan2(error_y, error_x)
            error_angular = self._normalize_angle(target_angle - self.robot.state.orientation)
        
        self.integral_angular += error_angular * dt
        derivative_angular = (error_angular - self.prev_error_angular) / dt
        
        angular_velocity = (
            self.kp_angular * error_angular +
            self.ki_angular * self.integral_angular +
            self.kd_angular * derivative_angular
        )
        
        # Update state
        self.prev_error_linear = error_linear
        self.prev_error_angular = error_angular
        self.prev_time = current_time
        
        return ControlInput(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=current_time
        )
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset(self) -> None:
        """Reset controller state."""
        self.prev_error_linear = 0.0
        self.prev_error_angular = 0.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.prev_time = 0.0


class PurePursuitController(RobotController):
    """
    Pure Pursuit controller for path following.
    
    Pure Pursuit is a geometric path following algorithm that
    calculates the steering angle to reach a lookahead point.
    """
    
    def __init__(self, robot: RobotModel, lookahead_distance: float = 1.0) -> None:
        """Initialize Pure Pursuit controller."""
        super().__init__(robot)
        self.lookahead_distance = lookahead_distance
    
    def compute_control(
        self, 
        target_position: Tuple[float, float],
        target_orientation: Optional[float] = None
    ) -> ControlInput:
        """
        Compute Pure Pursuit control input.
        
        Args:
            target_position: Target position
            target_orientation: Target orientation (ignored for Pure Pursuit)
            
        Returns:
            Control input
        """
        current_pos = self.robot.state.position
        current_orientation = self.robot.state.orientation
        
        # Vector to target
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        
        # Distance to target
        distance = math.sqrt(dx**2 + dy**2)
        
        # If close to target, stop
        if distance < 0.1:
            return ControlInput(linear_velocity=0.0, angular_velocity=0.0)
        
        # Calculate lookahead point
        lookahead_x = current_pos[0] + self.lookahead_distance * dx / distance
        lookahead_y = current_pos[1] + self.lookahead_distance * dy / distance
        
        # Calculate steering angle
        alpha = math.atan2(lookahead_y - current_pos[1], lookahead_x - current_pos[0]) - current_orientation
        alpha = self._normalize_angle(alpha)
        
        # Calculate velocities
        linear_velocity = min(self.robot.config.max_velocity, distance * 0.5)
        angular_velocity = 2 * linear_velocity * math.sin(alpha) / self.lookahead_distance
        
        return ControlInput(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=self.robot.state.timestamp
        )
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def create_robot(robot_type: str, config: Optional[RobotConfig] = None) -> RobotModel:
    """
    Factory function to create robot models.
    
    Args:
        robot_type: Type of robot ('differential_drive', 'omnidirectional')
        config: Robot configuration
        
    Returns:
        Robot model instance
    """
    config = config or RobotConfig()
    
    if robot_type.lower() == 'differential_drive':
        return DifferentialDriveRobot(config)
    elif robot_type.lower() == 'omnidirectional':
        return OmnidirectionalRobot(config)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def create_controller(controller_type: str, robot: RobotModel, **kwargs) -> RobotController:
    """
    Factory function to create robot controllers.
    
    Args:
        controller_type: Type of controller ('pid', 'pure_pursuit')
        robot: Robot model
        **kwargs: Controller-specific parameters
        
    Returns:
        Robot controller instance
    """
    if controller_type.lower() == 'pid':
        return PIDController(robot, **kwargs)
    elif controller_type.lower() == 'pure_pursuit':
        return PurePursuitController(robot, **kwargs)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def main() -> None:
    """Example usage of robot models and controllers."""
    import matplotlib.pyplot as plt
    
    # Create robot and controller
    robot = create_robot('differential_drive')
    controller = create_controller('pid', robot)
    
    # Set initial position
    robot.reset(position=(0.0, 0.0), orientation=0.0)
    
    # Simulate navigation to target
    target_position = (5.0, 5.0)
    trajectory = []
    
    for step in range(100):
        # Compute control
        control_input = controller.compute_control(target_position)
        
        # Apply control
        robot.apply_control(control_input, dt=0.1)
        
        # Record trajectory
        trajectory.append(robot.state.position)
        
        # Check if target reached
        distance = math.sqrt(
            (robot.state.position[0] - target_position[0])**2 + 
            (robot.state.position[1] - target_position[1])**2
        )
        
        if distance < 0.1:
            logger.info(f"Target reached in {step} steps")
            break
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    plt.scatter(*target_position, color='red', s=100, label='Target')
    plt.scatter(*trajectory[0], color='green', s=100, label='Start')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Navigation Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
