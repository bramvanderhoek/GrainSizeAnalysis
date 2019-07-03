"""preProcessing.py: some tools for doing preprocessing on OpenFOAM cases,
updating dictionaries and writing scripts to execute OpenFOAM cases on a computing cluster."""

import os
import numpy as np

def updateBlockMeshDict(path, xmin, xmax, ymin, ymax, height, mindist=None, nx=100, ny=100):
    """Replaces the minimum and maximum extents of the blockMeshDict found by the path according to xmin, xmax, ymin & ymax.
If mindist is provided, calculate the required amount of cells so that two cells fit within mindist diagonally,
otherwise nx and ny define the amount of cells.

PARAMETERS
----------
path : str
    Path to the folder where the blockMeshDict file is located.
xmin : float, int
    Minimum x-axis extent of the mesh.
xmax : float, int
    Maximum x-axis extent of the mesh.
ymin : float, int
    Minimum y-axis extent of the mesh.
ymax : float, int
    Maximum y-axis extent of the mesh.
height : float, int
    Height of the mesh (size along z-axis).
mindist : float, int
    Minimum distance between grains in the model.
nx : int
    Number of cells along x-axis if mindist is not used.
ny : int
    Number of cells along y-axis if mindist is not used."""

    bmd_old = open("{0}{1}blockMeshDict".format(path, os.sep), "r")
    bmd_new = open("{0}{1}blockMeshDict_new".format(path, os.sep), "w")

    if mindist:
        # Calculate required amount of cells in x and y direction
        cellsize = mindist / np.sqrt(8)
        nx = int(np.ceil((xmax - xmin) / cellsize))
        ny = int(np.ceil((ymax - ymin) / cellsize))

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
        bmd_new.write(line)

    bmd_old.close()
    bmd_new.close()

    os.replace("{0}{1}blockMeshDict_new".format(path, os.sep), "{0}{1}blockMeshDict".format(path, os.sep))

def updateSnappyHexMeshDict(path, stlFilename, locationInMesh, refinement=False, castellatedMesh=True, snap=True):
    """Update snappyHexMeshDict with new .stl filename and point in mesh.

PARAMETERS
----------
path : str
    Path to the folder where the snappyHexMeshDict is located.
stlFilename : str
    Filename of the stl file which will be encorporated into the snappyHexMeshDict.
locationInMesh : array_like
    Array of length 3 containing the coordinates to a random location inside of the mesh.
refinement : bool
    Whether or not refinement should be enabled in the snappyHexMeshDict.
castellatedMesh : bool
    Whether or not castellatedMesh step should be enabled in the snappyHexMeshDict.
snap : bool
    Whether or not the snap step should be enabled in the snappyHexMeshDict."""

    shmd_old = open("{0}{1}snappyHexMeshDict".format(path, os.sep), "r")
    shmd_new = open("{0}{1}snappyHexMeshDict_new".format(path, os.sep), "w")

    geometryFound = False
    refinementFound = False
    for line in shmd_old.readlines():
        if line.startswith("geometry"):
            geometryFound = True
        elif line.startswith("{") and geometryFound:
            pass
        elif geometryFound:
            line = "\t{0}.stl\n".format(stlFilename)
            geometryFound = False

        if line.startswith("castellatedMesh") and not line.startswith("castellatedMeshControls"):
            line = "castellatedMesh\t{0};\n".format("true" if castellatedMesh else "false")
        if line.startswith("snap") and line.split()[0] == "snap":
            line = "snap\t\t\t{0};\n".format("true" if snap else "false")

        if line.strip().startswith("refinementSurfaces"):
            refinementFound = True
        elif line.strip().startswith("level") and refinementFound:
            line = "\t\t\tlevel ({0} {0});\n".format(1 if refinement else 0)
            refinementFound = False

        if line.strip().startswith("locationInMesh"):
            line = "\tlocationInMesh ({0}e-3 {1}e-3 {2}e-3);\n".format(*[coord for coord in locationInMesh])
        shmd_new.write(line)

    shmd_old.close()
    shmd_new.close()

    os.replace("{0}{1}snappyHexMeshDict_new".format(path, os.sep), "{0}{1}snappyHexMeshDict".format(path, os.sep))

def updateDecomposeParDict(path, nCores):
    """Updates decomposeParDict with the appropriate amount of cores.

PARAMETERS
----------
path : str
    Path to the folder where the decomposeParDict is located.
nCores : int
    Number of cores the case will be decomposed into."""

    dpd_old = open("{0}{1}decomposeParDict".format(path, os.sep), "r")
    dpd_new = open("{0}{1}decomposeParDict_new".format(path, os.sep), "w")

    # Find a nice distribution of cores over x and y
    nx = int(np.ceil(np.sqrt(nCores)))
    while not nCores % nx == 0:
        nx += 1
    ny = nCores // nx

    for line in dpd_old.readlines():
        if line.startswith("numberOfSubdomains"):
            line = "numberOfSubdomains\t{0};\n".format(nCores)
        elif line.strip() and line.split()[0] == "n":
            line = "\tn\t\t\t\t({0} {1} 1);\n".format(nx, ny)
        dpd_new.write(line)

    dpd_old.close()
    dpd_new.close()

    os.replace("{0}{1}decomposeParDict_new".format(path, os.sep), "{0}{1}decomposeParDict".format(path, os.sep))

def updateExtrudeMeshDict(path, height):
    """Updates extrudeMesh dict with correct domain height.

PARAMETERS
----------
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

    os.replace("{0}{1}extrudeMeshDict_new".format(path.os.sep), "{0}{1}extrudeMeshDict".format(path, os.sep))

def createScriptHeader(nTasks, tasksPerNode, threadsPerCore, partition, name):
    """Creates a header for a bash script to be submitted to a cluster running Slurm.

PARAMETERS
----------
nTasks : int
    Maximum number of tasks to be launched by the script.
tasksPerNode : int
    Amount tasks to be invoked per computing node.
threadsPerCore : int
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
#SBATCH --job-name {4}\n\n""".format(nTasks, tasksPerNode, threadsPerCore, partition, name)

    return header

def createScriptModules(modules, scripts):
    """Creates the part of a bash script that loads modules and sources scripts.

PARAMETERS
----------
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).

RETURNS
-------
moduleString : str
    String to be added to bash script to load modules and source given scripts."""

    moduleString = ""
    for module in modules:
        moduleString += "module load {0}\n".format(module)
    for script in scripts:
        moduleString += "source {0}\n".format(script)
    moduleString += "\n"

    return moduleString


def createPreProcessingScript(caseName, nTasks, tasksPerNode, threadsPerCore, partition, modules, scripts, refinement=False):
    """Create a bash script to do pre-processing of a case.

PARAMETERS
----------
caseName : str
    Name of the case for which the bash script is created.
nTasks : int
    Maximum number of tasks to be launched by the script.
tasksPerNode : int
    Amount tasks to be invoked per computing node.
threadsPerCore : int
    Restrict node selection to nodes with at least the specified number of threads per core.
partition : str
    Which queue to submit the script to.
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).
refinement : bool
    Whether or not snappyHexMesh should run a refinement phase."""

    header = createScriptHeader(nTasks, tasksPerNode, threadsPerCore, partition, "{0}_pre".format(caseName))

    moduleString = createScriptModules(modules, scripts)

    commands = "blockMesh | tee blockMesh.log\n"

    if nTasks > 1:
        commands += """decomposePar
mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_0.log
reconstructParMesh
rm -rf processor*
cp -rf {1}/polyMesh constant/
rm -rf 1 2\n""".format(nTasks, "1" if refinement else "2")
    else:
        commands += "snappyHexMesh -overwrite | tee snappyHexMesh_0.log\n"

    commands += "extrudeMesh | tee extrudeMesh.log\n"

    # Write these commands to a first script if refinement is active
    if refinement:
        script = open("preprocessing_0.sh", "w")
        script.write(header)
        script.write(moduleString)
        script.write(commands)
        script.close()

        # Start a new set of commands that will be run after the first set if refinement is active
        commands = ""

        if nTasks > 1:
            commands += """decomposePar
mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_1.log
reconstructParMesh
rm -rf processor*
cp -rf 1/polyMesh constant/
rm -rf 1\n""".format(nTasks)
        else:
            commands += "snappyHexMesh -overwrite -parallel | tee snappyHexMesh_1.log\n"

    script = open("preprocessing{0}.sh".format("_1" if refinement else ""), "w")
    script.write(header)
    script.write(moduleString)
    script.write(commands)
    script.close()

def createSimulationScript(caseName, nTasks, tasksPerNode, threadsPerCore, partition, modules, scripts):
    """Create a bash script to run a prepared OpenFOAM case using simpleFoam solver and export to VTK.

PARAMETERS
----------
caseName : str
    Name of the case for which the bash script is created.
nTasks : int
    Maximum number of tasks to be launched by the script.
tasksPerNode : int
    Amount tasks to be invoked per computing node.
threadsPerCore : int
    Restrict node selection to nodes with at least the specified number of threads per core.
partition : str
    Which queue to submit the script to.
modules : list
    List of the names of modules to be loaded (as strings).
scripts : list
    List of the scripts to be sourced (as strings).
"""

    header = createScriptHeader(nTasks, tasksPerNode, threadsPerCore, partition, "{0}_sim".format(caseName))

    moduleString = createScriptModules(modules, scripts)

    if nTasks > 1:
        commands = """decomposePar
mpirun -np {0} simpleFoam -parallel | tee simpleFoam.log
reconstructPar
rm -rf processor*\n""".format(nTasks)
    else:
        commands = "simpleFoam | tee simpleFoam.log\n"

    commands += "foamToVTK -latestTime -ascii\n"

    script = open("runSimulations.sh", "w")
    script.write(header)
    script.write(moduleString)
    script.write(commands)
    script.close()