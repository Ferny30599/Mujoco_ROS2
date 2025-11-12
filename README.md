# Mujoco_ROS2
Simulation project integrating a MuJoCo model with ROS 2 through Python interfaces. Includes the XML model of dummy, STL meshes, communication bridge, and control module for real-time evaluation of exoskeleton based on driven cables and control strategies.

This repository contains the complete simulation model of dummy with generalized soft-exoskeleton based on driven cables developed in MuJoCo, along with the Python scripts used to interface the simulator with ROS 2 and the control module implemented for dynamic testing. The project is intended to provide a reproducible environment for evaluating the behavior of a robotic system through a high-fidelity physics simulation and a real-time communication bridge between MuJoCo and ROS 2.

Repository Contents
• MuJoCo XML Model

The XML file defines the robot’s structure, including its geometry, joints, sensors, actuators, and physical parameters.
It provides a coherent model compatible with the MuJoCo physics engine and suitable for dynamic simulations.

• STL Files

A collection of 3D mesh files required by the XML model.
These files allow accurate geometric rendering of the robot’s components and ensure correct spatial interpretation by the simulator.

• MuJoCo–ROS 2 Bridge (Python)

Python scripts implementing the communication interface between MuJoCo and ROS 2.
The bridge provides real-time exchange of joint states, control inputs (torques/commands), and any additional data needed for closed-loop operation.

• Control Module (Python)

The controller responsible for processing the system’s state feedback and generating the actuation commands applied during simulation.
This script handles ROS 2 subscriptions, control law computation, and command publication to the MuJoCo interface.

Project Objective

The purpose of this project is to offer an extensible and reproducible platform for the analysis, validation, and testing of control strategies applied to robotic systems simulated in MuJoCo and tightly integrated with the ROS 2 ecosystem.
It is designed for researchers, developers, and students who require a hybrid simulation-and-control environment with configurable and easily modifiable components.

Requirements

MuJoCo (version compatible with the provided XML model)

Python 3.x

ROS 2 (Humble, Iron, or the distribution used in your workspace)

Additional dependencies listed in requirements.txt or in the project’s documentation.
