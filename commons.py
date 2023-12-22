import argparse
from typing import Callable, Any
from argparse import Namespace


def make_printer(debug=False) -> Callable[..., Any]:
    """Make a dprint function that only logs if :debug: is True."""

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    return dprint


def parse_arguments() -> Namespace:
    """Get arguments for the program run."""

    parser = argparse.ArgumentParser(
        description="Calculate rod division to heat it to minimum desired temperature."
    )

    parser.add_argument(
        "--max-temperature", type=float, default=100.0, help="Maximum temperature"
    )
    parser.add_argument(
        "--rod-length", type=float, default=20.0, help="Length of the rod"
    )
    parser.add_argument(
        "--thermal-conductivity",
        type=float,
        default=401.0,
        help="Thermal conductivity coefficient",
    )
    parser.add_argument(
        "--desired-min-temperature",
        type=float,
        default=40.0,
        help="Desired minimum temperature",
    )
    parser.add_argument("--max-time", type=float, default=0.005, help="Maximum time")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--temperature-function-demo",
        action="store_true",
        help="Show time function demo",
    )
    parser.add_argument(
        "--optimisation",
        choices=["max_time", "max_temperature", "none"],
        default="none",
    )

    return parser.parse_args()
