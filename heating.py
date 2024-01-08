from numpy import sin, cos, pi, exp
from typing import Callable
from commons import make_printer

dprint = make_printer(debug=False)


# func should be (n: int) -> int
def calculate_infinite_series(func: Callable[[int], int], delta=1e-6) -> float:
    """
    Calculate the sum of a converging sequence.

    :param Callable[[int], int] func: function with one argument "n" that calculates the n-th member of the sequence
    :param delta: maximum difference between each next part of the sum to consider it calculated
    :return float: the calculated sum
    """

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


def find_function_argument_by_value(
    func, desired_value, arg_from, arg_to, max_iterations=1000, delta=1e-4
) -> float or None:
    """
    Find y^-1(x) given y and x.
    Only works for functions monotonous in the given interval.

    :param func: y from y^-1(x)
    :param desired_value: x from y^-1(x)
    :param arg_from: x left bound
    :param arg_to: x right bound
    :param max_iterations: hard limit on the search, will fail if y isn't monotonous on the interval
    :param delta: allowed difference from the :desired_value: for the found result
    :return float or None: the result of the y^-1(x) with :delta: precision or None if nothing found
    :raises ValueError: thrown if the bounds are invalid
    """

    if arg_from >= arg_to:
        raise ValueError("arg_from must be less than arg_to")

    function_slope = 1 if func(arg_to) - func(arg_from) > 0 else -1

    arg_delta = (arg_to - arg_from) / 2
    candidate_argument = arg_to - arg_delta
    counter = 0

    while counter < max_iterations:
        candidate_value = func(candidate_argument)
        if (
            candidate_value > desired_value
            and abs(candidate_value - desired_value) < delta
        ):
            dprint("solution found")
            return candidate_argument

        arg_delta /= 2

        dprint(f"idx {counter}: shorter, {arg_delta=}, {candidate_argument=}")

        if candidate_value < desired_value:
            candidate_argument += arg_delta * function_slope
        else:
            candidate_argument -= arg_delta * function_slope

        counter += 1

    return None


def find_max_division_length(
    max_time, rod_length, max_temperature, thermal_diffusivity, desired_min_temperature
) -> float:
    """
    Find max rod parts length to heat the whole rod to the desired temperature.

    :param float max_time: max possible time
    :param float rod_length: whole rod length
    :param float max_temperature: max time for the experiment
    :param float thermal_diffusivity: thermal diffusivity coefficient of the rod's material
    :param float desired_min_temperature: desired min temperature of the whole rod
    :return float: max rod parts length to reach the specified conditions
    """

    time_function = make_time_function(
        max_temperature, thermal_diffusivity, rod_length
    )
    max_heated_right_end_temperature = time_function(rod_length, max_time)
    if max_heated_right_end_temperature > desired_min_temperature:
        dprint("rod length is good enough")

        return rod_length

    return find_function_argument_by_value(
        lambda x: call_time_function_with_variable_config(
            max_temperature, thermal_diffusivity, x, x, max_time
        ),
        desired_min_temperature,
        1e-6,
        rod_length,
    )


def call_time_function_with_variable_config(
    max_temperature, thermal_diffusivity, rod_length, x, t
) -> float:
    """
    Create a time function for the specified conditions and call it on x, t

    :param float max_temperature: max possible temperature
    :param float thermal_diffusivity: thermal diffusivity coefficient of the rod's material
    :param float rod_length: whole rod length
    :param float x: position across the X axis of the rod
    :param float t: position in time
    :return float: temperature of the described rod at x, t
    """

    time_function = make_time_function(
        max_temperature, thermal_diffusivity, rod_length
    )

    return time_function(x, t)


def make_time_function(
    max_temperature, thermal_diffusivity, rod_length
) -> Callable[[float, float], float]:
    """
    Create a time function with the specified conditions.

    :param float max_temperature: max possible temperature
    :param float thermal_diffusivity: thermal diffusivity coefficient of the rod's material
    :param float rod_length: whole rod length
    :return Callable[[float, float], float]: time function of type u(t, x)
    """

    def time_function(x, t) -> float:
        if x < 0 or x > rod_length:
            raise Exception(f"x should be between 0 and L, got {x=}")

        def one_iteration(n):
            return (
                (cos(pi * (n + 0.5)) - 1)
                / (2 * n + 1.0)
                * sin(pi * (n + 0.5) / rod_length * x)
                * exp(-thermal_diffusivity * pi * (n + 0.5) / rod_length * t)
            )

        return max_temperature + 4.0 * max_temperature / pi * calculate_infinite_series(
            one_iteration
        )

    return time_function
