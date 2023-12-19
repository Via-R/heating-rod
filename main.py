from numpy import sin, cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def custom_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)


def custom_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)


# func should be (n: int) -> int
def calculate_infinite_series(func, delta=1e-6):
    def term_generator():
        n = 0
        current_term = func(n)
        yield current_term

        while True:
            n += 1
            next_term = func(n)
            yield next_term
            current_term = next_term

    terms = term_generator()
    series_sum = next(terms)

    prev_term = None
    for term in terms:
        if prev_term is not None and abs(term - prev_term) < delta:
            break
        series_sum += term
        prev_term = term

    return series_sum


# temp at x coordinate of the rod with length 'rod_length' at time t
def make_time_function(max_temperature, thermal_conductivity, rod_length):
    def time_function(x, t) -> float:
        if x < 0 or x > rod_length:
            raise Exception(f'x should be between 0 and L, got {x=}')

        def one_iteration(n):
            return (cos(pi * (n + 0.5)) - 1) / (2 * n + 1.) * sin(pi * (n + 0.5) / rod_length * x) * exp(
                -thermal_conductivity * pi * (n + 0.5) / rod_length * t)

        return max_temperature + 4. * max_temperature / pi * calculate_infinite_series(one_iteration)

    return time_function


def demo_display(max_temperature, rod_length, thermal_conductivity):
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
    frame_text = ax.text(0.7, 0.9, '', transform=ax.transAxes)

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
    plt.title('Temperature Distribution in a Rod over Time')
    plt.show()


def find_max_division_length(max_time, rod_length, max_temperature, thermal_conductivity, desired_min_temperature,
                             delta=1e-4):
    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)
    max_heated_right_end_temperature = time_function(rod_length, max_time)
    if max_heated_right_end_temperature > desired_min_temperature:
        print('rod length is good enough')
        return rod_length

    counter = 0

    delta_length = rod_length / 2
    division_length = rod_length - delta_length

    while counter < 100:
        max_heated_right_end_temperature = time_function(division_length, max_time)
        if max_heated_right_end_temperature > desired_min_temperature and abs(
                max_heated_right_end_temperature - desired_min_temperature) < delta:
            print('solution found')
            return division_length
        delta_length /= 2
        if max_heated_right_end_temperature < desired_min_temperature:
            print(f'shorter, {delta_length=}, {division_length=}')
            division_length -= delta_length
        else:
            print(f'longer, {delta_length=}, {division_length=}')
            division_length += delta_length

        counter += 1

    return None


def main():
    max_temperature = 100.0  # Maximum temperature
    rod_length = 10  # Length of the rod
    thermal_conductivity = 407  # thermal conductivity of 407 is copper
    # demo_display(max_temperature, rod_length, thermal_conductivity)
    # return

    desired_min_temperature = 40.0
    max_time = 0.005
    max_division_length = find_max_division_length(max_time, rod_length, max_temperature, thermal_conductivity,
                                                   desired_min_temperature)
    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)
    print(f'desired temperature: {desired_min_temperature}, calculated: {time_function(max_division_length, max_time)}')

    whole_divisions = np.ceil(rod_length / max_division_length)
    whole_division_length = rod_length / whole_divisions
    print(
        f'whole division result: {time_function(whole_division_length, max_time)}, division length: {whole_division_length}')

    # now minimise either max_temperature or max_time to get exactly desired_min_temperature with divison set as whole_division_length


if __name__ == '__main__':
    main()
