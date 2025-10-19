# Heating Rod Simulation

A Python-based simulation tool for analyzing heat distribution in a rod with multiple heating elements. The project solves the heat equation to determine optimal rod division and heating parameters.

## Overview

This project simulates heat diffusion in a rod that needs to be heated to a minimum desired temperature within a specified time constraint. The simulation uses the analytical solution to the one-dimensional heat equation and provides visualizations of temperature distribution over time.

### Physical Model

The simulation is based on the heat equation with the following assumptions:
- One-dimensional heat conduction along the rod
- Multiple heating elements placed symmetrically along the rod
- Each heating element heats a section of the rod from the center (maximum temperature)
- Temperature decreases towards the edges of each section

The temperature function `u(x, t)` is calculated using a Fourier series solution:

```
u(x, t) = T_max + (4 * T_max / π) * Σ [(cos(π(n + 0.5)) - 1) / (2n + 1) * sin(π(n + 0.5)x/L) * exp(-α * π(n + 0.5)t/L)]
```

Where:
- `T_max` = maximum temperature (at heating elements)
- `L` = rod length (or rod section length)
- `α` = thermal diffusivity coefficient
- `x` = position along the rod
- `t` = time

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heating-rod
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Numerical computations
- `matplotlib` - Visualization and animation
- `Pillow` - Image processing

## Usage

### Basic Simulation

Run the simulation with default parameters:
```bash
python main.py
```

### Custom Parameters

Specify custom rod properties and constraints:
```bash
python main.py --rod-length=0.3 --max-time=120 --desired-min-temperature=40 --thermal-diffusivity=0.000097
```

This command simulates an **aluminum rod** with:
- **Length**: 0.3 meters (30 cm)
- **Maximum heating time**: 120 seconds
- **Desired minimum temperature**: 40 K across the entire rod
- **Thermal diffusivity**: 0.000097 m²/s (typical for aluminum)

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-temperature` | float | 100.0 | Maximum temperature (K) at heating elements |
| `--rod-length` | float | 20.0 | Total length of the rod (meters) |
| `--thermal-diffusivity` | float | 401.0 | Thermal diffusivity coefficient (m²/s) |
| `--desired-min-temperature` | float | 40.0 | Minimum temperature required across the rod (K) |
| `--max-time` | float | 0.005 | Maximum time available for heating (seconds) |
| `--optimisation` | choice | none | Optimization mode: `max_time`, `max_temperature`, or `none` |
| `--temperature-function-demo` | flag | False | Show simple temperature function demonstration |
| `--debug` | flag | False | Enable debug output |

### Optimization Modes

1. **No Optimization (`none`)**: Uses maximum time and temperature
2. **Time Optimization (`max_time`)**: Minimizes heating time while maintaining desired temperature
3. **Temperature Optimization (`max_temperature`)**: Minimizes maximum temperature while staying within time constraint

Example with time optimization:
```bash
python main.py --rod-length=0.3 --max-time=120 --optimisation=max_time
```

### Temperature Function Demo

View a simple animation of heat propagation in a continuous rod:
```bash
python main.py --temperature-function-demo --max-temperature=100 --rod-length=0.3 --thermal-diffusivity=0.000097
```

### Key Functions

#### `heating.py`
- `make_time_function()`: Creates temperature function u(x, t)
- `calculate_infinite_series()`: Computes Fourier series sum
- `find_max_division_length()`: Determines optimal rod section length
- `find_function_argument_by_value()`: Binary search for inverse function values

#### `demo.py`
- `rod_temperature_demo()`: Main animation with multiple heating elements
- `temperature_function_demo()`: Simple continuous rod heating demo
- `rod_division_demo()`: Complete workflow with calculations and visualization
- `print_end_conditions()`: Formatted output of results

## Material Properties (Thermal Diffusivity)

Common thermal diffusivity values at room temperature:

| Material | Thermal Diffusivity (m²/s) |
|----------|----------------------------|
| Aluminum | 9.7 × 10⁻⁵ (0.000097) |
| Copper | 1.11 × 10⁻⁴ (0.000111) |
| Steel | 1.2 × 10⁻⁵ (0.000012) |
| Brass | 3.4 × 10⁻⁵ (0.000034) |
| Iron | 2.3 × 10⁻⁵ (0.000023) |

## Examples

### Example 1: Fast Heating of Copper Rod
```bash
python main.py --rod-length=0.5 --max-time=60 --desired-min-temperature=50 --thermal-diffusivity=0.000111 --max-temperature=150
```

### Example 2: Slow Heating of Steel Rod
```bash
python main.py --rod-length=1.0 --max-time=300 --desired-min-temperature=60 --thermal-diffusivity=0.000012 --max-temperature=200
```

### Example 3: Optimize Temperature for Aluminum
```bash
python main.py --rod-length=0.3 --max-time=120 --desired-min-temperature=40 --thermal-diffusivity=0.000097 --optimisation=max_temperature
```

## Limitations

- Assumes ideal heat conduction (no heat loss to environment)
- One-dimensional model (ignores radial temperature variations)
- Assumes symmetric heating from center of each section
- Requires monotonic functions for optimization algorithms
- Convergence threshold for infinite series may affect accuracy
