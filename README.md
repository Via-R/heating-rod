# <p align="center">1D Rod heating task solver ∙ ![Windows support](https://badgen.net/badge/icon/Windows?icon=windows&label) ![MacOS support](https://badgen.net/badge/icon/MacOS?icon=apple&label) ![Linux](https://badgen.net/badge/icon/Linux?icon=gnome&label)</p>

<p align="center"><img src="images/process-100" alt="Solution example"/></p>

This is a demo project which emulates a solution for a 1D rod heating problem. It is made as a console tool to solve the task of heating a perfectly insulated rod with a heating element until its minimum temperature reaches a preset threshold. There are two optimisations available: for the maximum time and the heating element temperature. Both textual and visual representations of the solution are given.

# Math base

The given task of heating a 1D rod is solved using Fourier heat equation with boundary and initial conditions. The temperature function is explicitly represented in the source code, and it's used to show an animation of heat distribution during the simulation. A method of dichotomy is used to calculate arguments needed for the optimised answers as described before.

# Running the project

In order to run this project, you need to install the requirements and run the `main.py` file with parameters that suit your needs. If you want to find main thermal diffusivity values, they are available [here](https://en.wikipedia.org/wiki/Thermal_diffusivity#Thermal_diffusivity_of_selected_materials_and_substances).

## Creating virtual environment (optional)

Best practice to run such projects as this it to have a virtual environment set up, I offer using the simplest version, `virtualenv`. You can find instructions on how to set it up online, depending on your platform. If you already have it, run the following command:

    $ virtualenv venv
    
Linux / MacOS:

    $ source venv/bin/activate

Windows:
 
    > venv\Scripts\activate

Once you're done with running this project, you can deactivate the environment:

    $ deactivate
    
## Installing requirements

    $ pip install -r requirements.txt

## Launching code

I advice you to read `help` on your first launch, there you will find a description of all arguments you will need to properly run the script:

    $ python main.py -h

Example:

    $ python main.py --thermal-diffusivity=9.7e-05 --max-time=1200 --rod-length=50
