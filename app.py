"""
System Integration Assignment 1 - Last Question Solver

This module provides solutions for the final question of Assignment 1 in System's Integration.

The code performs the following tasks:
1. Imports necessary libraries for numerical computation and optimization.
2. Defines a Transfer Function representing a system.
3. Implements a dynamic system model using ordinary differential equations.
4. Defines an objective function for Particle Swarm Optimization (PSO) based parameter identification.
5. Plots various system responses, including impulse and step responses, and a Pole-Zero Plot.
6. Performs PSO optimization to identify system parameters.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse, step, freqresp
from scipy.integrate import odeint
from pyswarm import pso

# Define the system coefficients and create a Transfer Function
numerator = [13, 9]
denominator = [1, 5, 17, 11]
system = TransferFunction(numerator, denominator)


def model(y, t, params):
    """
    Dynamic system model for ordinary differential equations.
    """
    a3, a2, a1, b3, b2, b1 = params
    dydt = [y[1], y[2], -(a1*y[2] + a2*y[1] + a3*y[0]) +
            b1*y[2] + b2*y[1] + b3*y[0]]
    return dydt


def objective(params, t, actual_response):
    """
    Objective function for PSO-based parameter identification.
    """
    initial_conditions = [0, 0, 0]  # Initial conditions for y, y', and y''
    predicted_response = odeint(model, initial_conditions, t, args=(params,))
    difference = actual_response - predicted_response[:, 0]
    obj_value = np.sum(difference**2)
    return obj_value


def plot_system_responses(system):
    """
    Plot various responses of the system, including impulse, step, and Pole-Zero Plot.
    """
    # Impulse Response
    time_imp, response_imp = impulse(system)
    plt.figure()
    plt.plot(time_imp * 33, response_imp * 33)  # Scale time and response
    plt.title('Impulse Response (Scaled)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    # Step Response
    time_step, response_step = step(system)
    plt.figure()
    plt.plot(time_step, response_step * 2.5)  # Scale response only
    plt.title('Step Response (Scaled)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    # Pole-Zero Plot
    poles, zeros = system.poles, system.zeros
    plt.figure()
    plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Pole-Zero Plot')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.show()


def plot_bode_plot(system):
    """
    Plot Bode plot - Magnitude and Phase.
    """
    freq, response = freqresp(system)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, 20 * np.log10(np.abs(response)))
    plt.title('Bode Plot - Magnitude')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2, 1, 2)
    plt.semilogx(freq, np.angle(response, deg=True))
    plt.title('Bode Plot - Phase')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')
    plt.show()


# Plot various system responses
plot_system_responses(system)

# Plot Bode plot
plot_bode_plot(system)

# System Identification using PSO
# Generate example data (replace with actual data)
t = np.linspace(0, 10, 100)
actual_response = odeint(model, [0, 0, 0], t, args=([1, 5, 17, 11, 13, 9],))

# PSO Optimization
lb, ub = [-10]*6, [10]*6  # Bounds for parameters
best_params, obj_value = pso(
    objective, lb, ub, args=(t, actual_response[:, 0]))
print("Identified Parameters:", best_params)
print("Final Objective Function Value:", obj_value)
