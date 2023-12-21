import numpy as np

from commons import parse_arguments
from demo import rod_division_demo, temperature_function_demo


def main():
    args = parse_arguments()

    if args.temperature_function_demo:
        temperature_function_demo(
            args.max_temperature, args.rod_length, args.thermal_conductivity
        )
        return

    rod_division_demo(
        args.max_temperature,
        args.rod_length,
        args.thermal_conductivity,
        args.desired_min_temperature,
        args.max_time,
    )


def test():
    rod_part_x_values = np.linspace(0, 2.5, 5)
    print(np.flip(rod_part_x_values))
    print(np.concatenate((rod_part_x_values, np.flip(rod_part_x_values))))
    x_values = np.tile(np.concatenate(rod_part_x_values, np.flip(rod_part_x_values)), 3)
    print(x_values)


if __name__ == "__main__":
    # TODO:
    #  1) add animation of heating the rod with calculated divisions
    #  2) add typing and black formatter
    main()
    # test()
