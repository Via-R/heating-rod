import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from heating import make_time_function, find_max_division_length, find_function_argument_by_value, \
    call_time_function_with_variable_config


def print_end_conditions(conditions_name, max_temperature, max_time, desired_min_temperature, estimated_max_temperature,
                         estimated_min_temperature, estimated_time, estimated_division_length, divisions):
    print('*' * 12 + ' ' + conditions_name + ' ' + '*' * 12)
    print(
        f'Limitations: \n\nMax temperature: {max_temperature}º, min temperature: {desired_min_temperature}º, time: {max_time} units\n')
    print(
        f'Estimations: \n\nMax temperature: {estimated_max_temperature}º, min temperature: {estimated_min_temperature}º, time: {estimated_time} units')
    print(f'Rod division length: {estimated_division_length:.2f}, divisions: {int(divisions * 2)}')
    print('*' * 36, end='\n\n')


# temp at x coordinate of the rod with length 'rod_length' at time t
def temperature_function_demo(max_temperature, rod_length, thermal_conductivity):
    # Parameters
    time_from = 0
    time_to = 0.1
    time_iterations = 100
    num_points = 500  # Number of points along the rod
    delta_temperature = 0.001 * max_temperature  # Homogeneity threshold

    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)

    # Create initial temperature distribution along the rod
    x_values = np.linspace(0, rod_length, num_points)
    initial_temperatures = []  # Temperature at time t=0

    # Populate initial temperatures one point at a time
    for x in x_values:
        initial_temp_at_x = time_function(x, 0)  # Temperature at time t=0 for each point
        initial_temperatures.append(initial_temp_at_x)

    # Create figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot(x_values, initial_temperatures, color='red')
    frame_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")

    iteration_num = 0

    # Update function for animation
    def update(frame):
        nonlocal iteration_num

        current_temperatures = np.array([time_function(x_value, frame) for x_value in x_values.tolist()])
        current_temperatures[current_temperatures < 0] = 0

        # Update the plot with new temperatures
        line.set_ydata(current_temperatures.tolist())
        frame_text.set_text(f'Time passed: {iteration_num}/{time_iterations}')

        # Check for homogeneity to stop the animation
        if (np.max(np.abs(current_temperatures - max_temperature)) < delta_temperature
                or iteration_num == time_iterations):
            ani.event_source.stop()  # Stop animation when homogeneous

        iteration_num += 1

        return line, frame_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.linspace(time_from, time_to, time_iterations).tolist(), interval=100,
                        blit=True)

    # Set plot parameters
    plt.xlabel('Position along the rod')
    plt.ylabel('Temperature')
    plt.title(f'Temperature distribution, max temp: {max_temperature:.2f}')
    plt.show()


# temp at x coordinate of the rod with length 'rod_length' at time t
def rod_temperature_demo(max_temperature, rod_length, thermal_conductivity, heaters_amount, rod_part_length, graph_max_temperature=None,
                         time_from=0, time_to=0.1, time_iterations=100, rod_part_points_amount=500):
    # Parameters
    time_function = make_time_function(max_temperature, thermal_conductivity, rod_part_length)

    desired_temperature = time_function(rod_part_length, time_to)

    # Create initial temperature distribution along the rod
    rod_part_x_values = np.linspace(0, rod_part_length, rod_part_points_amount)
    rod_part_temperatures = np.array([time_function(x, 0) for x in rod_part_x_values])

    x_values = np.linspace(0, rod_length, heaters_amount * 2 * rod_part_points_amount)
    rod_temperatures = np.tile(
        np.concatenate(
            (np.flip(rod_part_temperatures), rod_part_temperatures)
        ),
        heaters_amount)

    # Create figure and axis for the plot
    fig, ax = plt.subplots()
    ax.set_ylim([0, graph_max_temperature or max_temperature])
    line, = ax.plot(x_values, rod_temperatures, color='red')
    ax.plot(x_values, [desired_temperature] * heaters_amount * 2 * rod_part_points_amount, color='grey')
    frame_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")

    iteration_num = -1

    # Update function for animation
    def update(frame):
        nonlocal iteration_num

        current_rod_part_temperatures = np.array([time_function(x, frame) for x in rod_part_x_values])
        current_rod_temperatures = np.tile(
            np.concatenate(
                (np.flip(current_rod_part_temperatures), current_rod_part_temperatures)
            ),
            heaters_amount)

        current_rod_temperatures[current_rod_temperatures < 0] = 0

        # Update the plot with new temperatures
        line.set_ydata(current_rod_temperatures.tolist())

        if iteration_num > -1:
            frame_text.set_text(f'Time passed: {iteration_num}/{time_iterations}')

        # Check for homogeneity to stop the animation
        if iteration_num == time_iterations:
            ani.event_source.stop()  # Stop animation when homogeneous

        iteration_num += 1

        return line, frame_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.linspace(time_from, time_to, time_iterations).tolist(), interval=100,
                        blit=True)

    # Set plot parameters
    plt.xlabel('Position along the rod')
    plt.ylabel('Temperature')
    plt.title(f'Temperature distribution, max temp: {max_temperature:.2f}, max time: {time_to:.5f}')
    plt.show()


def rod_division_demo(max_temperature, rod_length, thermal_conductivity, desired_min_temperature, max_time):
    half_rod_length = rod_length / 2
    max_division_length = find_max_division_length(max_time, half_rod_length, max_temperature, thermal_conductivity,
                                                   desired_min_temperature)
    if not max_division_length:
        print("Verdict: Impossible to heat the rod in these conditions")
        return

    whole_divisions = int(np.ceil(half_rod_length / max_division_length))
    whole_division_length = half_rod_length / whole_divisions

    time_function = make_time_function(max_temperature, thermal_conductivity, whole_division_length)

    print_end_conditions('Regular solution',
                         max_temperature=max_temperature,
                         max_time=max_time,
                         desired_min_temperature=desired_min_temperature,
                         estimated_max_temperature=max_temperature,
                         estimated_min_temperature=time_function(whole_division_length, max_time),
                         estimated_time=max_time,
                         estimated_division_length=whole_division_length,
                         divisions=whole_divisions)

    optimal_time = find_function_argument_by_value(lambda x: time_function(whole_division_length, x),
                                                   desired_min_temperature, 0, max_time)
    if not optimal_time:
        print("Verdict: Couldn't optimise time")
        return

    print('We can optimise either by time or by max temperature:')
    print_end_conditions('Optimised time',
                         max_temperature=max_temperature,
                         max_time=max_time,
                         desired_min_temperature=desired_min_temperature,
                         estimated_max_temperature=max_temperature,
                         estimated_min_temperature=time_function(whole_division_length, optimal_time),
                         estimated_time=optimal_time,
                         estimated_division_length=whole_division_length,
                         divisions=whole_divisions)

    optimal_max_temperature = find_function_argument_by_value(
        lambda x: call_time_function_with_variable_config(x, thermal_conductivity, whole_division_length,
                                                          whole_division_length, max_time),
        desired_min_temperature, 0, max_temperature)

    print_end_conditions('Optimised max temperature',
                         max_temperature=max_temperature,
                         max_time=max_time,
                         desired_min_temperature=desired_min_temperature,
                         estimated_max_temperature=optimal_max_temperature,
                         estimated_min_temperature=call_time_function_with_variable_config(optimal_max_temperature,
                                                                                           thermal_conductivity,
                                                                                           whole_division_length,
                                                                                           whole_division_length,
                                                                                           max_time),
                         estimated_time=max_time,
                         estimated_division_length=whole_division_length,
                         divisions=whole_divisions)

    print('Verdict: all optimisations successful')

    rod_temperature_demo(max_temperature=optimal_max_temperature,
                         rod_length=rod_length,
                         thermal_conductivity=thermal_conductivity,
                         heaters_amount=whole_divisions,
                         rod_part_length=whole_division_length,
                         time_to=max_time,
                         graph_max_temperature=max_temperature)
