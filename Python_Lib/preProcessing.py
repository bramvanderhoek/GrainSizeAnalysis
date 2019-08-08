"""preProcessing.py: some tools for doing preprocessing on OpenFOAM cases,
updating dictionaries and writing scripts to execute OpenFOAM cases on a computing cluster."""

import os
import numpy as np

def update_blockMeshDict(path, domain): # mindist=None, nx=100, ny=100):
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
            cell_size : int, float
                cell size of block mesh to start with (mm).
    """

    xmax, xmin, ymax, ymin, zmax = (
        domain["xmax"],
        domain["xmin"],
        domain["ymax"],
        domain["ymin"],
        domain["cell_size"]
    )
    
    bmd_old = open("{}/blockMeshDict".format(path), "r")
    bmd_new = open("{}/blockMeshDict_new".format(path), "w")

    nx = int(np.ceil((xmax - xmin) / domain["cell_size"]))
    ny = int(np.ceil((ymax - ymin) / domain["cell_size"]))

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
            line = "z_max\t{0};\n".format(zmax)
        elif line.startswith("nx"):
            line = "nx\t{0};\n".format(nx)
        elif line.startswith("ny"):
            line = "ny\t{0};\n".format(ny)
        elif line.strip().startswith("top"):
            top_found = True
        elif top_found and line.strip().startswith("type"):
            if (
                line.strip().split()[-1] == "cyclic;"
                or line.strip().split()[-1] == "cyclicAMI;"
            ):
                top_cyclic = True
        elif top_found and top_cyclic and line.strip().startswith("separationVector"):
            line = "\t\tseparationVector (0 -{0}e-3 0);\n".format(y_dist)
            top_found = False
        elif line.strip().startswith("bottom"):
            bottom_found = True
        elif bottom_found and line.strip().startswith("type"):
            if (
                line.strip().split()[-1] == "cyclic;"
                or line.strip().split()[-1] == "cyclicAMI;"
            ):
                bottom_cyclic = True
        elif (
            bottom_found
            and bottom_cyclic
            and line.strip().startswith("separationVector")
        ):
            line = "\t\tseparationVector (0 {0}e-3 0);\n".format(y_dist)
            bottom_found = False

        bmd_new.write(line)

    bmd_old.close()
    bmd_new.close()

    os.replace(
        "{0}{1}blockMeshDict_new".format(path, os.sep),
        "{0}{1}blockMeshDict".format(path, os.sep),
    )


def update_snappyHexMeshDict(
    path,
    stl_filename,
    cell_size,
    location_in_mesh,
    refinement=False,
    snap=True,
    castellated_mesh=True,
):

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
        Whether or not the snap step should be enabled in the snappyHexMeshDict.
    """

    shmd_old = open("{}/snappyHexMeshDict".format(path), "r")
    shmd_new = open("{}/snappyHexMeshDict_new".format(path), "w")

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

        if line.startswith("castellatedMesh") and not line.startswith(
            "castellatedMeshControls"
        ):
            line = "castellatedMesh\t{0};\n".format(
                "true" if castellated_mesh else "false"
            )
        if line.startswith("snap") and line.split()[0] == "snap":
            line = "snap\t\t\t{0};\n".format("true" if snap else "false")

        if line.strip().startswith("refinementSurfaces"):
            refinement_found = True
        elif line.strip().startswith("level") and refinement_found:
            line = "\t\t\tlevel ({0} {0});\n".format(1 if refinement else 0)
            refinement_found = False

        if line.strip().startswith("locationInMesh"):
            line = "\tlocationInMesh ({0}e-3 {1}e-3 {2}e-3);\n".format(
                *[coord for coord in location_in_mesh]
            )

        if line.strip().startswith("minVol") and not line.strip().startswith(
            "minVolRatio"
        ):
            # Set minimum volume to a fraction of expected cell volume
#            line = "\tminVol\t{0};\n".format(cell_size ** 2 * height * 0.0001)
            line = "\tminVol\t{0};\n".format( (cell_size  * 0.001)**3 * 0.0001)
        shmd_new.write(line)

    shmd_old.close()
    shmd_new.close()

    os.replace(
        "{}/snappyHexMeshDict_new".format(path),
        "{}/snappyHexMeshDict".format(path),
    )

def update_decomposeParDict(path, cores_per_sim=1):
    """Updates decomposeParDict with the appropriate amount of cores.

    PARAMETERS
    ----------
    path : str
        Path to the folder where the decomposeParDict is located.
    cores_per_sim : int
        Number of cores the case will be decomposed into.
    """

    dpd_old = open("{0}{1}decomposeParDict".format(path, os.sep), "r")
    dpd_new = open("{0}{1}decomposeParDict_new".format(path, os.sep), "w")

    # Find a nice distribution of cores over x and y
    nx = int(np.ceil(np.sqrt(cores_per_sim)))
    while not cores_per_sim % nx == 0:
        nx += 1
    ny = cores_per_sim // nx

    for line in dpd_old.readlines():
        if line.startswith("numberOfSubdomains"):
            line = "numberOfSubdomains\t{0};\n".format(cores_per_sim)
        elif line.strip() and line.split()[0] == "n":
            line = "\tn\t\t\t\t({0} {1} 1);\n".format(nx, ny)
        dpd_new.write(line)

    dpd_old.close()
    dpd_new.close()

    os.replace(
        "{}/decomposeParDict_new".format(path),
        "{}/decomposeParDict".format(path),
    )


def update_extrudeMeshDict(path, height):
    """Updates extrudeMesh dict with correct domain height.

    PARAMETERS
    ----------
    path : str
        Path to the folder in which the extrudeMeshDict file is located.
    height : float, int
        Height of the model (z-axis).
    """

    emd_old = open("{}/extrudeMeshDict".format(path), "r")
    emd_new = open("{}/extrudeMeshDict_new".format(path), "w")

    for line in emd_old.readlines():
        if line.startswith("thickness"):
            line = "thickness\t{0};\n".format(height * 0.001)
        emd_new.write(line)

    emd_old.close()
    emd_new.close()

    os.replace(
        "{}/extrudeMeshDict_new".format(path),
        "{}/extrudeMeshDict".format(path),
    )


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
    if not os.path.isfile("tail {}".format(log_file)):
        status = False
    else:

        last_word = os.popen("tail {}".format(log_file)).read().split()[-1]   
        # If log file ends with the word 'End', we know that the process ended properly, otherwise something went wrong
        if last_word == "End" or last_word == "run":
            status = True
        else:
            status = False

    return status
