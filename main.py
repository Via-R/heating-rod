from commons import parse_arguments
from demo import rod_division_demo, temperature_function_demo


def main():
    args = parse_arguments()

    if args.temperature_function_demo:
        temperature_function_demo(args.max_temperature, args.rod_length, args.thermal_conductivity)
        return

    rod_division_demo(args.max_temperature, args.rod_length, args.thermal_conductivity, args.desired_min_temperature,
                      args.max_time)


if __name__ == '__main__':
    main()
