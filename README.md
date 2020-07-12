# A data-driven approach to systematically and reproduciably optimize force-field parameters 

<img src="workflow.png" width="800">

## Features:
* Efficient adoption of parallism to perform sampling with a parallel simulator
* Diverse choices of force-field functional forms supported by the chosen simulator
* Flexible inclusions of distinct physical properties as reference data
* Modular design to facilitate the exstensions with user-defined:
    - objective functional forms
    - sampling methods/force-field potential functional forms
    - optimization algorithms

## Softwares required:

* A compiled MD/MC packages exectuable (LAMMPS is already supported)

* Slurm Workload Manager (or equivalent) 

* Python 3.7 (Numpy 1.18.1) 

* pytest 5.4.2

* Intel Fortran compiler (> version 18.0.3).
  gfortran (4.8.5) also compiles successfully,
  but the program may not work with a long absolute file path.

## Installation on Linux:

conda is the recommended package manager. If 'conda' is not not found or root previliage is required,
you can download anaconda: https://www.anaconda.com/products/individual to your home directory.
Then, install the package: https://docs.anaconda.com/anaconda/install/  

* create a conda environment with specific version of numpy and python:  

```
conda create -n "env name" python=3.7 numpy=1.18.1 
```

* copy this GitHub repo to your local directory:

```
git clone https://github.com/jingxiangguo/Python-force-field-parameterization-workflow.git 
```

```
cd Python-force-field-parameterization-workflow
``` 
* install the package to your conda environment (Two options)

```
pip install .
```
* or you can install the package in an editable mode (For furture development):

```
pip install -e .  
```
* A Fortran library "fortranAPI" come with the package.
Inside Fortran routines are C-interoperable, and thus can be
callable through Python using ctypes modules.

* Note that this Fortran library is independent. 
It provides some trajectory readers 
and radial distribution function calculations routines.

To compile it, run GNU "make" command. 

``` 
make
``` 
Test your installation.

``` 
optimize -h
``` 
if you see a list of command line argument options, then your installation is successful.

Now, run some unit-testings to further check the installation

```
pytest optimizer/

pytest fortranAPI/
```

## Examples:

## References: 

[1]: Chan, H., Cherukara, M. J., Narayanan, B., Loeffler, T. D., Benmore, C., Gray, S. K., & Sankaranarayanan, S. K. R. S. (2019). Machine learning coarse grained models for water. Nature Communications, 10(1), 379. https://doi.org/10.1038/s41467-018-08222-6 

[2]: Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Computational Optimization and Applications, 51(1), 259–277. https://doi.org/10.1007/s10589-010-9329-3   

[3]: Wang, L.P., Chen, J., & Van Voorhis, T. (2013). Systematic Parametrization of Polarizable Force Fields from Quantum Chemistry Data. Journal of Chemical Theory and Computation, 9(1), 452–460. https://doi.org/10.1021/ct300826t  

[4]: Ercolessi, F., & Adams, J. B. (1994). Interatomic Potentials from First-Principles Calculations: The Force-Matching Method. Europhysics Letters ({EPL}), 26(8), 583–588. https://doi.org/10.1209/0295-5075/26/8/005  

[5]: Sundararaman, S., Huang, L., Ispas, S., & Kob, W. (2018). New optimization scheme to obtain interaction potentials for oxide glasses. Journal of Chemical Physics, 148(19). https://doi.org/10.1063/1.5023707 
