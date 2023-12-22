from commons import parse_arguments
from demo import rod_division_demo, temperature_function_demo


def main():
    """Program handler."""

    args = parse_arguments()

    if args.temperature_function_demo:
        temperature_function_demo(
            args.max_temperature, args.rod_length, args.thermal_conductivity
        )
        return

    rod_division_demo(
        max_temperature=args.max_temperature,
        rod_length=args.rod_length,
        thermal_conductivity=args.thermal_conductivity,
        desired_min_temperature=args.desired_min_temperature,
        max_time=args.max_time,
        optimisation=args.optimisation,
    )


if __name__ == "__main__":
    main()
