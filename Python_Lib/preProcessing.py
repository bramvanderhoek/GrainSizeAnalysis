"""preProcessing.py: some tools for doing preprocessing on OpenFOAM cases,
updating dictionaries and writing scripts to execute OpenFOAM cases on a computing cluster."""

import os
import numpy as np


def update_blockMeshDict(path, domain, mindist=None, nx=100, ny=100):
    """Replaces the minimum and maximum extents of the blockMeshDict found by the path according to xmin, xmax, ymin & ymax.
If mindist is provided, calculate the required amount of cells so that two cells fit within mindist diagonally,
otherwise nx and ny define the amount of cells.

PARAMETERS
----------
path : str
    Path to the folder where the blockMeshDict file is located.
domain : dict
    Dictionary containing parameters of the modelling domain.

    Mandatory keywords:
        xmin : int, float
            Lower value of the domain size along the x-axis (mm).
        xmax : int, float
            Upper value of the domain size along the x-axis (mm).
        ymin : int, float
            Lower value of the domain size along the y-axis (mm).
        ymax : int, float
            Upper value of the domain size along the y-axis (mm).
        height : int, float
            Height of the domain (i.e. its thickness along the z-axis) (mm).

mindist : float, int
    Minimum distance between grains in the model.
nx : int
    Number of cells along x-axis if mindist is not used.
ny : int
    Number of cells along y-axis if mindist is not used."""

    xmax, xmin, ymax, ymin = domain["xmax"], domain["xmin"], domain["ymax"], domain["ymin"]
    height = domain["height"]

    bmd_old = open("{0}{1}blockMeshDict".format(path, os.sep), "r")
    bmd_new = open("{0}{1}blockMeshDict_new".format(path, os.sep), "w")

    if mindist:
        # Calculate required amount of cells in x and y direction
        cellsize = mindist / np.sqrt(8)
        nx = int(np.ceil((xmax - xmin) / cellsize))
        ny = int(np.ceil((ymax - ymin) / cellsize))

    y_dist = ymax - ymin

    top_found = False
    bottom_found = False
    top_cyclic = False
    bottom_cyclic = False
    for line in bmd_old.readlines():
        if line.startswith("x_min"):
            line = "x_min\t{0};\n".format(xmin)
        elif line.startswith("x_max"):
            line = "x_max\t{0};\n".format(xmax)
        elif line.startswith("y_min"):
            line = "y_min\t{0};\n".format(ymin)
        elif line.startswith("y_max"):
            line = "y_max\t{0};\n".format(ymax)
        elif line.startswith("z_min"):
            line = "z_min\t0;\n"
        elif line.startswith("z_max"):
            line = "z_max\t{0};\n".format(height)
        elif line.startswith("nx"):
            line = "nx\t{0};\n".format(nx)
        elif line.startswith("ny"):
            line = "ny\t{0};\n".format(ny)
        elif line.strip().startswith("top"):
            top_found = True
        elif top_found and line.strip().startswith("type"):
            if line.strip().split()[-1] == "cyclic;" or line.strip().split()[-1] == "cyclicAMI;":
                top_cyclic = True
        elif top_found and top_cyclic and line.strip().startswith("separationVector"):
            line = "\t\tseparationVector (0 -{0}e-3 0);\n".format(y_dist)
            top_found = False
        elif line.strip().startswith("bottom"):
            bottom_found = True
        elif bottom_found and line.strip().startswith("type"):
            if line.strip().split()[-1] == "cyclic;" or line.strip().split()[-1] == "cyclicAMI;":
                bottom_cyclic = True
        elif bottom_found and bottom_cyclic and line.strip().startswith("separationVector"):
            line = "\t\tseparationVector (0 {0}e-3 0);\n".format(y_dist)
            bottom_found = False

        bmd_new.write(line)

    bmd_old.close()
    bmd_new.close()

    os.replace("{0}{1}blockMeshDict_new".format(path, os.sep), "{0}{1}blockMeshDict".format(path, os.sep))


def update_snappyHexMeshDict(path, stl_filename, height, mindist, location_in_mesh, refinement=False, castellated_mesh=True, snap=True):
    """Update snappyHexMeshDict with new .stl filename and point in mesh.

PARAMETERS
----------
path : str
    Path to the folder where the snappyHexMeshDict is located.
stl_filename : str
    Filename of the stl file which will be incorporated into the snappyHexMeshDict.
height : int, float
    Height of the domain (i.e. its thickness along the z-axis) (mm).
mindist : float, int
    Minimum distance between grains in the model (mm).
location_in_mesh : array_like
    Array of length 3 containing the coordinates to a random location inside of the mesh.
refinement : bool
    Whether or not refinement should be enabled in the snappyHexMeshDict.
castellated_mesh : bool
    Whether or not castellatedMesh step should be enabled in the snappyHexMeshDict.
snap : bool
    Whether or not the snap step should be enabled in the snappyHexMeshDict."""

    # Convert to meters
    height = height * 0.001
    mindist = mindist * 0.001

    # Calculate approximate minimum cell size along x and y-dimensions
    cellsize = mindist / np.sqrt(8)

    shmd_old = open("{0}{1}snappyHexMeshDict".format(path, os.sep), "r")
    shmd_new = open("{0}{1}snappyHexMeshDict_new".format(path, os.sep), "w")

    geometry_found = False
    refinement_found = False
    for line in shmd_old.readlines():
        if line.startswith("geometry"):
            geometry_found = True
        elif line.startswith("{") and geometry_found:
            pass
        elif geometry_found:
            line = "\t{0}.stl\n".format(stl_filename)
            geometry_found = False

        if line.startswith("castellatedMesh") and not line.startswith("castellatedMeshControls"):
            line = "castellatedMesh\t{0};\n".format("true" if castellated_mesh else "false")
        if line.startswith("snap") and line.split()[0] == "snap":
            line = "snap\t\t\t{0};\n".format("true" if snap else "false")

        if line.strip().startswith("refinementSurfaces"):
            refinement_found = True
        elif line.strip().startswith("level") and refinement_found:
            line = "\t\t\tlevel ({0} {0});\n".format(1 if refinement else 0)
            refinement_found = False

        if line.strip().startswith("locationInMesh"):
            line = "\tlocationInMesh ({0}e-3 {1}e-3 {2}e-3);\n".format(*[coord for coord in location_in_mesh])

        if line.strip().startswith("minVol") and not line.strip().startswith("minVolRatio"):
            # Set minimum volume to a fraction of expected cell volume
            line = "\tminVol\t{0};\n".format(cellsize**2 * height * 0.0001)
        shmd_new.write(line)

    shmd_old.close()
    shmd_new.close()

    os.replace("{0}{1}snappyHexMeshDict_new".format(path, os.sep), "{0}{1}snappyHexMeshDict".format(path, os.sep))


def update_decomposeParDict(path, n_cores):
    """Updates decomposeParDict with the appropriate amount of cores.

PARAMETERS
----------
path : str
    Path to the folder where the decomposeParDict is located.
n_cores : int
    Number of cores the case will be decomposed into."""

    dpd_old = open("{0}{1}decomposeParDict".format(path, os.sep), "r")
    dpd_new = open("{0}{1}decomposeParDict_new".format(path, os.sep), "w")

    # Find a nice distribution of cores over x and y
    nx = int(np.ceil(np.sqrt(n_cores)))
    while not n_cores % nx == 0:
        nx += 1
    ny = n_cores // nx

    for line in dpd_old.readlines():
        if line.startswith("numberOfSubdomains"):
            line = "numberOfSubdomains\t{0};\n".format(n_cores)
        elif line.strip() and line.split()[0] == "n":
            line = "\tn\t\t\t\t({0} {1} 1);\n".format(nx, ny)
        dpd_new.write(line)

    dpd_old.close()
    dpd_new.close()

    os.replace("{0}{1}decomposeParDict_new".format(path, os.sep), "{0}{1}decomposeParDict".format(path, os.sep))


def update_extrudeMeshDict(path, height):
    """Updates extrudeMesh dict with correct domain height.

PARAMETERS
----------
path : str
    Path to the folder in which the extrudeMeshDict file is located.
height : float, int
    Height of the model (z-axis)."""

    emd_old = open("{0}{1}extrudeMeshDict".format(path, os.sep), "r")
    emd_new = open("{0}{1}extrudeMeshDict_new".format(path, os.sep), "w")

    for line in emd_old.readlines():
        if line.startswith("thickness"):
            line = "thickness\t{0};\n".format(height*0.001)
        emd_new.write(line)

    emd_old.close()
    emd_new.close()

    os.replace("{0}{1}extrudeMeshDict_new".format(path, os.sep), "{0}{1}extrudeMeshDict".format(path, os.sep))


def create_script_header(n_tasks, tasks_per_node, threads_per_core, partition, name):
    """Creates a header for a bash script to be submitted to a cluster running Slurm.

PARAMETERS
----------
n_tasks : int
    Maximum number of tasks to be launched by the script.
tasks_per_node : int
    Amount tasks to be invoked per computing node.
threads_per_core : int
    Restrict node selection to nodes with at least the specified number of threads per core.
partition : str
    Which queue to submit the script to.
name : str
    Name of the job.

RETURNS
-------
header : str
    A header to be put at the top of a batch script to be submitted to a cluster running Slurm."""

    header = """#!/bin/bash
#SBATCH --ntasks={0}
#SBATCH --ntasks-per-node={1}
#SBATCH --threads-per-core={2}
#SBATCH --partition={3}
#SBATCH -o {4}.%N.%j.out
#SBATCH -e {4}.%N.%j.err
#SBATCH --job-name {4}\n\n""".format(n_tasks, tasks_per_node, threads_per_core, partition, name)

    return header


def create_script_modules(modules, scripts):
    """Creates the part of a bash script that loads modules and sources scripts.

PARAMETERS
----------
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).

RETURNS
-------
module_string : str
    String to be added to bash script to load modules and source given scripts."""

    module_string = ""
    for module in modules:
        module_string += "module load {0}\n".format(module)
    for script in scripts:
        module_string += "source {0}\n".format(script)
    module_string += "\n"

    return module_string


def create_pre_processing_script(case_name, n_tasks, tasks_per_node, threads_per_core, partition, modules, scripts, refinement=False):
    """Create a bash script to do pre-processing of a case.

PARAMETERS
----------
case_name : str
    Name of the case for which the bash script is created.
n_tasks : int
    Maximum number of tasks to be launched by the script.
tasks_per_node : int
    Amount tasks to be invoked per computing node.
threads_per_core : int
    Restrict node selection to nodes with at least the specified number of threads per core.
partition : str
    Which queue to submit the script to.
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).
refinement : bool
    Whether or not snappyHexMesh should run a refinement phase."""

    header = create_script_header(n_tasks, tasks_per_node, threads_per_core, partition, "{0}_pre".format(case_name))

    module_string = create_script_modules(modules, scripts)

    commands = "blockMesh | tee blockMesh.log\n"

    if n_tasks > 1:
        commands += """decomposePar
mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_0.log
reconstructParMesh
rm -rf processor*
cp -rf {1}/polyMesh constant/
rm -rf 1 2\n""".format(n_tasks, "1" if refinement else "2")
    else:
        commands += "snappyHexMesh -overwrite | tee snappyHexMesh_0.log\n"

    commands += "extrudeMesh | tee extrudeMesh.log\n"

    # Write these commands to a first script if refinement is active
    if refinement:
        script = open("preprocessing_0.sh", "w")
        script.write(header)
        script.write(module_string)
        script.write(commands)
        script.close()

        # Start a new set of commands that will be run after the first set if refinement is active
        commands = ""

        if n_tasks > 1:
            commands += """decomposePar
mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_1.log
reconstructParMesh
rm -rf processor*
cp -rf 1/polyMesh constant/
rm -rf 1\n""".format(n_tasks)
        else:
            commands += "snappyHexMesh -overwrite -parallel | tee snappyHexMesh_1.log\n"

    script = open("preprocessing{0}.sh".format("_1" if refinement else ""), "w")
    script.write(header)
    script.write(module_string)
    script.write(commands)
    script.close()


def create_simulation_script(case_name, n_tasks, tasks_per_node, threads_per_core, partition, modules, scripts):
    """Create a bash script to run a prepared OpenFOAM case using simpleFoam solver and export to VTK.

PARAMETERS
----------
case_name : str
    Name of the case for which the bash script is created.
n_tasks : int
    Maximum number of tasks to be launched by the script.
tasks_per_node : int
    Amount tasks to be invoked per computing node.
threads_per_core : int
    Restrict node selection to nodes with at least the specified number of threads per core.
partition : str
    Which queue to submit the script to.
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).
"""

    header = create_script_header(n_tasks, tasks_per_node, threads_per_core, partition, "{0}_sim".format(case_name))

    module_string = create_script_modules(modules, scripts)

    if n_tasks > 1:
        commands = """decomposePar
mpirun -np {0} simpleFoam -parallel | tee simpleFoam.log
reconstructPar
rm -rf processor*\n""".format(n_tasks)
    else:
        commands = "simpleFoam | tee simpleFoam.log\n"

    commands += "foamToVTK -latestTime -ascii\n"

    script = open("runSimulations.sh", "w")
    script.write(header)
    script.write(module_string)
    script.write(commands)
    script.close()


def check_log(log_file):
    """Checks OpenFOAM log file to see if an OpenFOAM process ended properly or aborted due to an error.
Returns True if log ended properly, else returns False.

PARAMETERS
----------
log_file : str
    Path to the log file to be checked.

RETURNS
-------
status : bool
    True or False value depending on whether or not the OpenFOAM process ended properly, respectively."""

    # Get the last word from the log file using the 'tail' command
    last_word = os.popen("tail {0}".format(log_file)).read().split()[-1]

    # If log file ends with the word 'End', we know that the process ended properly, otherwise something went wrong
    if last_word == "End" or last_word == "run":
        status = True
    else:
        status = False

    return status
