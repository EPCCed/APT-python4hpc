# Practical 1: Exporing Python Performance with Computational Fluid Dynamics

- Connect to ARCHER2 or Cirrus with X forwarding turned on (`ssh -XY`) 
    - You are free to use a machine of your choosing other than ARCHER2 or Cirrus, however it will need the following:
        - a Python distribution (Python 2 or Python 3) that includes NumPy, SciPy, and Numba (e.g. the Anaconda distribution)
        - gcc and/or gfortran

- Clone this repository: `git clone https://github.com/EPCCed/APT-python4hpc.git`

## ARCHER2
- `module load cray-python`
- `pip install --user matplotlib` (once only, to be able to plot `flow.png` files)
- `module swap PrgEnv-cray PrgEnv-gnu` (needed for `ctypes` or `f2py` part of practical) 
- `export PATH=/home/d180/shared/imagemagick/bin:$PATH` (needed to make the `display` command available to view `flow.png` files - alternatively download and view these locally). 
- `pip install --user numba` (once only, to be able to use Numba)
- Follow the practical instructions in the browser, starting with [README.ipynb](https://github.com/EPCCed/APT-python4hpc/blob/master/exercises/01-cfd/README.ipynb)

## Cirrus
- `module load anaconda gcc/8.2.0 ImageMagick`
- To view `flow.png` files, use the `display` command (need to load the `ImageMagick` module for this to work), alternatively download and view these locally. 
- Follow the practical instructions in the browser, starting with [README.ipynb](https://github.com/EPCCed/APT-python4hpc/blob/master/exercises/01-cfd/README.ipynb)
