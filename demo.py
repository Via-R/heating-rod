import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from heating import (
    make_time_function,
    find_max_division_length,
    find_function_argument_by_value,
    call_time_function_with_variable_config,
)

DEMO_WINDOW_NAME = "Heating rod"


def print_end_conditions(
    conditions_name,
    rod_length,
    max_temperature,
    max_time,
    desired_min_temperature,
    thermal_diffusivity,
    estimated_max_temperature,
    estimated_min_temperature,
    estimated_time,
    estimated_division_length,
    divisions,
) -> None:
    """
    Print out given initial conditions and one calculated solution.

    :param str conditions_name: name for the given solution
    :param float rod_length: whole rod length (initial condition)
    :param float max_temperature: max temperature (initial condition)
    :param float max_time: max time (initial condition)
    :param float desired_min_temperature: desired minimal temperature across the rod (initial condition)
    :param float thermal_diffusivity: thermal diffusivity coefficient of the given rod's material
    :param float estimated_max_temperature: max temperature reached during the experiment
    :param float estimated_min_temperature: min rod temperature at the end of the experiment
    :param float estimated_time: time passed to reach the desired state
    :param float estimated_division_length: resulting rod parts length
    :param int divisions: amount of rod parts / heating elements
    """

    print("*" * 12 + " " + conditions_name + " " + "*" * 12)
    print(
        f"Initial conditions: \n\nMax temperature: {max_temperature}º K\n"
        f"Min temperature: {desired_min_temperature}º K\n"
        f"Time: {max_time} seconds\n"
        f"Whole rod length: {rod_length} meters\n"
        f"Thermal diffusivity: {thermal_diffusivity} m^2/s\n"
    )
    print("-" * 24)
    print(
        f"Estimations:\n\nMax temperature: {estimated_max_temperature}º K\n"
        f"Min temperature: {estimated_min_temperature}º K\n"
        f"Time: {estimated_time} seconds\n"
        f"Rod parts length: {estimated_division_length:.2f} meters, parts/heating elements: {int(divisions)}"
    )
    print("*" * 36, end="\n\n")


def temperature_function_demo(max_temperature, rod_length, thermal_diffusivity) -> None:
    """
    Show an animation of how temperature changes in the rod with time.

    :param float max_temperature: max possible temperature
    :param float rod_length: whole rod length
    :param float thermal_diffusivity: thermal diffusivity coefficient of the rod's material
    """

    time_from = 0
    time_to = 0.3
    time_iterations = 200
    num_points = 500  # Number of points along the rod
    delta_temperature = 0.001 * max_temperature  # Homogeneity threshold

    time_function = make_time_function(max_temperature, thermal_diffusivity, rod_length)

    # Create initial temperature distribution along the rod
    x_values = np.linspace(0, rod_length, num_points)
    initial_temperatures = [time_function(x, 0) for x in x_values]

    fig, ax = plt.subplots(num=DEMO_WINDOW_NAME)
    (line,) = ax.plot(
        x_values, initial_temperatures, color="red", label="actual temperature"
    )
    frame_text = ax.text(
        0.7,
        0.85,
        "",
        transform=ax.transAxes,
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        ha="center",
    )

    iteration_num = 0

    def update(frame):
        nonlocal iteration_num

        current_temperatures = np.array(
            [time_function(x_value, frame) for x_value in x_values.tolist()]
        )
        current_temperatures[current_temperatures < 0] = 0

        line.set_ydata(current_temperatures.tolist())
        frame_text.set_text(f"Time iterations: {iteration_num}/{time_iterations}")

        # Check for homogeneity to stop the animation
        if (
            np.max(np.abs(current_temperatures - max_temperature)) < delta_temperature
            or iteration_num == time_iterations
        ):
            frame_text.set_text("Rod is fully heated")
            ani.event_source.stop()

        iteration_num += 1

        return line, frame_text

    ani = FuncAnimation(
        fig,
        update,
        frames=np.linspace(time_from, time_to, time_iterations).tolist(),
        interval=100,
        blit=True,
    )

    plt.xlabel("Position along the rod, m")
    plt.ylabel("Temperature, ºK")
    plt.title(f"Temperature distribution, max temp: {max_temperature:.2f}")
    plt.legend(loc="lower right")
    plt.show()


def rod_temperature_demo(
    max_temperature,
    rod_length,
    thermal_diffusivity,
    heaters_amount,
    rod_part_length,
    desired_min_temperature,
    graph_max_temperature=None,
    time_from=0,
    time_to=0.1,
    time_iterations=100,
    rod_part_points_amount=500,
) -> None:
    """
    Show an animation of how the rod is heated with singular heating elements.

    :param float max_temperature: max temperature across the rod
    :param float rod_length: whole rod length
    :param float thermal_diffusivity: thermal diffusivity coefficient
    :param int heaters_amount: amount of rod parts / heating elements
    :param float rod_part_length: rod parts length
    :param float desired_min_temperature: desired min temperature of the whole rod
    :param float or None graph_max_temperature: y scale size (to show if temperature didn't reach its max value)
    :param float time_from: emulation start time
    :param float time_to: emulation end time
    :param int time_iterations: amount of time points
    :param int rod_part_points_amount: amount of points in the rod parts (x values to calculate the temperatures for
        the graph)
    """

    time_function = make_time_function(
        max_temperature, thermal_diffusivity, rod_part_length
    )

    rod_part_x_values = np.linspace(0, rod_part_length, rod_part_points_amount)
    rod_part_temperatures = np.array([time_function(x, 0) for x in rod_part_x_values])

    x_values = np.linspace(0, rod_length, heaters_amount * 2 * rod_part_points_amount)
    rod_temperatures = np.tile(
        np.concatenate((np.flip(rod_part_temperatures), rod_part_temperatures)),
        heaters_amount,
    )

    fig, ax = plt.subplots(num=DEMO_WINDOW_NAME)
    ax.set_ylim([0, graph_max_temperature or max_temperature])
    (line,) = ax.plot(
        x_values, rod_temperatures, color="black", label="actual temperature"
    )
    ax.plot(
        x_values,
        [desired_min_temperature] * heaters_amount * 2 * rod_part_points_amount,
        color="gray",
        linestyle="--",
        label="desired temperature threshold",
    )
    frame_text = ax.text(
        0.7,
        0.85,
        "",
        transform=ax.transAxes,
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        ha="center",
    )

    iteration_num = -1

    def update(frame):
        nonlocal iteration_num

        current_rod_part_temperatures = np.array(
            [time_function(x, frame) for x in rod_part_x_values]
        )
        current_rod_temperatures = np.tile(
            np.concatenate(
                (np.flip(current_rod_part_temperatures), current_rod_part_temperatures)
            ),
            heaters_amount,
        )

        current_rod_temperatures[current_rod_temperatures < 0] = 0

        line.set_ydata(current_rod_temperatures.tolist())

        if iteration_num > -1:
            frame_text.set_text(f"Time passed: {iteration_num}/{time_iterations}")

        # if iteration_num in [0, 33, 66, 100]:
        #     fig.savefig(f"images/process-{iteration_num}.png", dpi=150, bbox_inches="tight")

        if iteration_num == time_iterations:
            ani.event_source.stop()

        iteration_num += 1

        return line, frame_text

    ani = FuncAnimation(
        fig,
        update,
        frames=np.linspace(time_from, time_to, time_iterations).tolist(),
        interval=100,
        blit=True,
    )

    plt.xlabel("Position along the rod", fontsize=16)
    plt.ylabel("Temperature", fontsize=16)
    plt.title(
        f"Temperature distribution, max temp: {max_temperature:.2f}, max time: {time_to:.5f}"
    )
    plt.legend(loc="lower right")
    plt.show()


def rod_division_demo(
    max_temperature,
    rod_length,
    thermal_diffusivity,
    desired_min_temperature,
    max_time,
    optimisation,
) -> None:
    """
    Start all calculations and animations demo.

    :param float max_temperature: max possible temperature for the rod
    :param float rod_length: whole rod length
    :param float thermal_diffusivity: thermal diffusivity coefficient of the rod's material
    :param float desired_min_temperature: desired min temperature across the rod
    :param float max_time: max possible time for the process
    :param str optimisation: selected type of optimisations: max_time, max_temperature or none
    """

    half_rod_length = rod_length / 2
    max_division_length = find_max_division_length(
        max_time,
        half_rod_length,
        max_temperature,
        thermal_diffusivity,
        desired_min_temperature,
    )
    if not max_division_length:
        print("Verdict: Impossible to heat the rod in these conditions")
        return

    whole_divisions = int(np.ceil(half_rod_length / max_division_length))
    whole_division_length = half_rod_length / whole_divisions

    time_function = make_time_function(
        max_temperature, thermal_diffusivity, whole_division_length
    )

    print_end_conditions(
        "Regular solution",
        rod_length=rod_length,
        max_temperature=max_temperature,
        max_time=max_time,
        desired_min_temperature=desired_min_temperature,
        thermal_diffusivity=thermal_diffusivity,
        estimated_max_temperature=max_temperature,
        estimated_min_temperature=time_function(whole_division_length, max_time),
        estimated_time=max_time,
        estimated_division_length=whole_division_length * 2,
        divisions=whole_divisions,
    )

    if optimisation == "max_time":
        optimal_time = find_function_argument_by_value(
            lambda x: time_function(whole_division_length, x),
            desired_min_temperature,
            0,
            max_time,
        )
        if not optimal_time:
            print("Verdict: Couldn't optimise time")
            return

        print("We can optimise either by time or by max temperature:")
        print_end_conditions(
            "Optimised time",
            rod_length=rod_length,
            max_temperature=max_temperature,
            max_time=max_time,
            desired_min_temperature=desired_min_temperature,
            thermal_diffusivity=thermal_diffusivity,
            estimated_max_temperature=max_temperature,
            estimated_min_temperature=time_function(
                whole_division_length, optimal_time
            ),
            estimated_time=optimal_time,
            estimated_division_length=whole_division_length * 2,
            divisions=whole_divisions,
        )

        rod_temperature_demo(
            max_temperature=max_temperature,
            rod_length=rod_length,
            thermal_diffusivity=thermal_diffusivity,
            heaters_amount=whole_divisions,
            rod_part_length=whole_division_length,
            desired_min_temperature=desired_min_temperature,
            time_to=optimal_time,
            graph_max_temperature=max_temperature,
        )

    elif optimisation == "max_temperature":
        optimal_max_temperature = find_function_argument_by_value(
            lambda x: call_time_function_with_variable_config(
                x,
                thermal_diffusivity,
                whole_division_length,
                whole_division_length,
                max_time,
            ),
            desired_min_temperature,
            0,
            max_temperature,
        )

        print_end_conditions(
            "Optimised max temperature",
            rod_length=rod_length,
            max_temperature=max_temperature,
            max_time=max_time,
            desired_min_temperature=desired_min_temperature,
            thermal_diffusivity=thermal_diffusivity,
            estimated_max_temperature=optimal_max_temperature,
            estimated_min_temperature=call_time_function_with_variable_config(
                optimal_max_temperature,
                thermal_diffusivity,
                whole_division_length,
                whole_division_length,
                max_time,
            ),
            estimated_time=max_time,
            estimated_division_length=whole_division_length * 2,
            divisions=whole_divisions,
        )

        rod_temperature_demo(
            max_temperature=optimal_max_temperature,
            rod_length=rod_length,
            thermal_diffusivity=thermal_diffusivity,
            heaters_amount=whole_divisions,
            rod_part_length=whole_division_length,
            desired_min_temperature=desired_min_temperature,
            time_to=max_time,
            graph_max_temperature=max_temperature,
        )

    else:
        print("No optimisations requested")

        rod_temperature_demo(
            max_temperature=max_temperature,
            rod_length=rod_length,
            thermal_diffusivity=thermal_diffusivity,
            heaters_amount=whole_divisions,
            rod_part_length=whole_division_length,
            desired_min_temperature=desired_min_temperature,
            time_to=max_time,
            graph_max_temperature=max_temperature,
        )
