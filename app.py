"""
This module is used for solving the last Question of Assisgenment 1 of System's Integration.
We First import all the necessary imports.
The first section is used for ploting the bode plot and the responses of the system.
The second section is for PSO.
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
    This is the model function
    """
    a3, a2, a1, b3, b2, b1 = params
    dydt = [y[1], y[2], -(a1*y[2] + a2*y[1] + a3*y[0]) +
            b1*y[2] + b2*y[1] + b3*y[0]]
    return dydt


def objective(params, t, actual_response):
    """
    This function returns the objective value
    """
    initial_conditions = [0, 0, 0]  # Initial conditions for y, y', and y''
    predicted_response = odeint(model, initial_conditions, t, args=(params,))
    difference = actual_response - predicted_response[:, 0]
    obj_value = np.sum(difference**2)
    return obj_value
# Plot Impulse and Step Responses, and Pole-Zero Plot


def plot_system_responses(system):
    """
    This function is used to plot all the relevant responses
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


plot_system_responses(system)


def plot_bode_plot(system):
    """
    This function is used to plot the Bode plot
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
