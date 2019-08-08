#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:25:50 2019

@author: zech0001
"""
import os

def create_script_header(
    name,
    cores_per_sim=1,
    tasks_per_node=1,
    threads_per_core=2,
    partition="allq", 
):
    """Creates a header for a bash script to be submitted to a cluster running Slurm.

    PARAMETERS
    ----------
    cores_per_sim : int
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
#SBATCH --job-name {4}\n\n""".format(
        cores_per_sim, tasks_per_node, threads_per_core, partition, name
    )

    return header


def create_script_modules(
    modules=["opt/all", "gcc/6.4.0", "openmpi/gcc-6.4.0/3.1.2", "openFoam/6"],
    scripts=["/trinity/opt/apps/software/openFoam/version6/OpenFOAM-6/etc/bashrc"],
):
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


def create_pre_processing_script(
    path,
    case_name,
    cores_per_sim=1,
    refinement=False,
    cluster=True,
    **kwargs
):
    """Create a bash script to do pre-processing of a case.

    PARAMETERS
    ----------
        case_name : str
            Name of the case for which the bash script is created.
        cores_per_sim : int
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
            Whether or not snappyHexMesh should run a refinement phase.
    """

    if cluster:
        header = create_script_header("{0}_preprocessing".format(case_name),**kwargs)
        module_string = create_script_modules(**kwargs)
    else:
        header,module_string='#!/bin/bash\n',''

    commands = "blockMesh | tee blockMesh.log\n"

    if cores_per_sim > 1:
        commands += """decomposePar
        mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_0.log
        reconstructParMesh
        rm -rf processor*
        cp -rf {1}/polyMesh constant/
        rm -rf 1 2\n""".format(
            cores_per_sim, "1" if refinement else "2"
        )
    else:
        commands += "snappyHexMesh -overwrite | tee snappyHexMesh_0.log\n"

    commands += "extrudeMesh | tee extrudeMesh.log\n"

    # Write these commands to a first script if refinement is active
    if refinement:
        script = open(path + "/preprocessing_0.sh", "w")
        script.write(header)
        script.write(module_string)
        script.write(commands)
        script.close()

        # Start a new set of commands that will be run after the first set if refinement is active
        commands = ""

        if cores_per_sim > 1:
            commands += """decomposePar
            mpirun -np {0} snappyHexMesh -parallel | tee snappyHexMesh_1.log
            reconstructParMesh
            rm -rf processor*
            cp -rf 1/polyMesh constant/
            rm -rf 1\n""".format(
                cores_per_sim
            )
        else:
            commands += "snappyHexMesh -overwrite -parallel | tee snappyHexMesh_1.log\n"

    script = open("{}preprocessing{}.sh".format(path ,"_1" if refinement else ""), "w")
    script.write(header)
    script.write(module_string)
    script.write(commands)
    script.close()

    print("Preprocessing script successfully created: \n {}".format("{}preprocessing{}.sh".format(path ,"_1" if refinement else "")))


def create_simulation_script(
    path,
    case_name,
    cores_per_sim=1,
    cluster=True,
    **kwargs
):
    """Create a bash script to run a prepared OpenFOAM case using simpleFoam solver and export to VTK.

    PARAMETERS
    ----------
    case_name : str
        Name of the case for which the bash script is created.
    cores_per_sim : int
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

    if cluster:
        header = create_script_header("{0}_sim".format(case_name),**kwargs)
        module_string = create_script_modules(**kwargs)
    else:
        header, module_string = "#!/bin/bash \n", ""

    if cores_per_sim > 1:
        commands = """decomposePar
    mpirun -np {0} simpleFoam -parallel | tee simpleFoam.log
    reconstructPar
    rm -rf processor*\n""".format(
            cores_per_sim
        )
    else:
        commands = "simpleFoam | tee simpleFoam.log\n"

    commands += "foamToVTK -latestTime -ascii\n"

    script = open("{}runSimulations.sh".format(path), "w")
    script.write(header)
    script.write(module_string)
    script.write(commands)
    script.close()

    print("Simulation script successfully created: \n {}".format("{}runSimulations.sh".format(path)))

def create_post_processing_script(
        case_dir, 
        case_name, 
        script_path, 
        vtk_file,
        kin_visc=10 ** -6, 
        density=1000, 
        margin=0, 
        **kwargs
        ):
    """Create a bash script that can be run to do post processing on a completed OpenFOAM case.

    PARAMETERS
    ----------
    case_dir : str
        Path to case directory.
    case_name : str
        Name of the OpenFOAM case.
    script_path : str
        Path to the python script that will do the post-processing.
    vtk_file : str
        Path to the VTK file to be analysed.
    kin_visc : float, int
        Kinematic viscosity of the fluid in the simulation.
    density : float, int
        Density of the fluid in the simulation.
    margin : float, int
        Margin of the model that should not be taken into account when calculating porosity and permeability."""

    header = create_script_header( "{0}_post".format(case_name), **kwargs)

    modules = "module load anaconda3/2019.03\n\n"

    commands = "python3 {0} {1} {2} {3} {4}".format(script_path, case_dir, vtk_file, kin_visc, density, margin)

    script = open("{}postprocessing.sh".format(case_dir), "w")
    script.write(header)
    script.write(modules)
    script.write(commands)
    script.close()

    print("Postprocessing script successfully created: \n {}".format("{}postprocessing.sh".format(case_dir)))

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
   
    if not os.path.isfile(log_file):
        status = False
    else:
        last_word = os.popen("tail {}".format(log_file)).read().split()[-1]   
        # If log file ends with the word 'End', we know that the process ended properly, otherwise something went wrong
        if last_word == "End" or last_word == "run":
            status = True
        else:
            status = False

    return status

def check_file(file,i=0,process='Process',overwrite=False,log=False):

    if os.path.isfile(file):
        if overwrite:
            print(" {} for case {} already performed \n \
                    Simulation repeated, output overwritten.".format(process,i))
            status=False
        else:
            if log:
                last_word = os.popen("tail {}".format(file)).read().split()[-1]   
                # If log file ends with the word 'End', we know that the process ended properly, otherwise something went wrong
                if last_word == "End" or last_word == "run":
                    status = True
                    print(" {} for case {} performed properly  \
                                \n continue with next case.".format(process,i))
                else:
                    status = False
                    print("{} of case {} failed to run properly.".format(process,i))
            else:
                status=True
                print(" {} for case {} already performed  \
                            \n continue with next case.".format(process,i))        
    else:
        status=False
    
    return status
