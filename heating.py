import numpy as np
from numpy import sin, cos, pi, exp

from commons import make_printer

dprint = make_printer(debug=False)


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


def make_time_function(max_temperature, thermal_conductivity, rod_length):
    def time_function(x, t) -> float:
        if x < 0 or x > rod_length:
            raise Exception(f'x should be between 0 and L, got {x=}')

        def one_iteration(n):
            return (cos(pi * (n + 0.5)) - 1) / (2 * n + 1.) * sin(pi * (n + 0.5) / rod_length * x) * exp(
                -thermal_conductivity * pi * (n + 0.5) / rod_length * t)

        return max_temperature + 4. * max_temperature / pi * calculate_infinite_series(one_iteration)

    return time_function
