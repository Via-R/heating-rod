from numpy import sin, cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


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


# func should be a closure with one argument only (the one that can change), only works for monotonous functions (or monotonous values between arg_from and arg_to)
def find_function_argument_by_value(func, desired_value, arg_from, arg_to, max_iterations=1000, delta=1e-4):
    if arg_from >= arg_to:
        raise Exception('arg_from must be less than arg_to')
    function_slope = 1 if func(arg_from + delta) - func(arg_from) > 0 else -1

    arg_delta = (arg_to - arg_from) / 2
    candidate_argument = arg_to - arg_delta
    counter = 0

    while counter < max_iterations:
        candidate_value = func(candidate_argument)
        if candidate_value > desired_value and abs(candidate_value - desired_value) < delta:
            dprint('solution found')
            return candidate_argument

        arg_delta /= 2

        dprint(f'idx {counter}: shorter, {arg_delta=}, {candidate_argument=}')

        if candidate_value < desired_value:
            candidate_argument += arg_delta * function_slope
        else:
            candidate_argument -= arg_delta * function_slope

        counter += 1

    return None


def find_max_division_length(max_time, rod_length, max_temperature, thermal_conductivity, desired_min_temperature):
    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)
    max_heated_right_end_temperature = time_function(rod_length, max_time)
    if max_heated_right_end_temperature > desired_min_temperature:
        dprint('rod length is good enough')

        return rod_length

    return find_function_argument_by_value(lambda x: time_function(x, max_time), desired_min_temperature, 0, rod_length)


def call_time_function_with_variable_max_temp(max_temperature, thermal_conductivity, rod_length, x, t):
    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)

    return time_function(x, t)


def print_end_conditions(conditions_name, max_temperature, max_time, desired_min_temperature, estimated_max_temperature,
                         estimated_min_temperature, estimated_time, estimated_division_length, divisions):
    print('*' * 12 + ' ' + conditions_name + ' ' + '*' * 12)
    print(
        f'Limitations: \n\nMax temperature: {max_temperature}º, min temperature: {desired_min_temperature}º, time: {max_time} units\n')
    print(
        f'Estimations: \n\nMax temperature: {estimated_max_temperature}º, min temperature: {estimated_min_temperature}º, time: {estimated_time} units')
    print(f'Rod division length: {estimated_division_length}, divisions: {divisions}')
    print('*' * 36, end='\n\n')


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
    if not max_division_length:
        print("Verdict: Impossible to heat the rod in these conditions")
        return

    time_function = make_time_function(max_temperature, thermal_conductivity, rod_length)

    whole_divisions = np.ceil(rod_length / max_division_length)
    whole_division_length = rod_length / whole_divisions

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
        lambda x: call_time_function_with_variable_max_temp(x, thermal_conductivity, whole_division_length,
                                                            whole_division_length, max_time),
        desired_min_temperature, 0, max_temperature)

    print_end_conditions('Optimised max temperature',
                         max_temperature=max_temperature,
                         max_time=max_time,
                         desired_min_temperature=desired_min_temperature,
                         estimated_max_temperature=optimal_max_temperature,
                         estimated_min_temperature=call_time_function_with_variable_max_temp(optimal_max_temperature,
                                                                                             thermal_conductivity,
                                                                                             whole_division_length,
                                                                                             whole_division_length,
                                                                                             max_time),
                         estimated_time=max_time,
                         estimated_division_length=whole_division_length,
                         divisions=whole_divisions)

    print('Verdict: all optimisations successful')


if __name__ == '__main__':
    main()
