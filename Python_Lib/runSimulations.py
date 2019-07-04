import os
import time
import numpy as np
import subprocess

import randomCircles
import postProcessing as postP
import preProcessing as preP

########################################
####    GRAIN SIZE DISTRIBUTION     ####
########################################

# Dictionary defining the model for the grain size distribution
model = dict(distribution_type="truncLogNormal",
             rmin=0.05,
             rmax=0.8,
             rmean=0.35,
             rstd=0.25,
             mindist=0.025,
             seed=False)

############################
####    DOMAIN SIZE     ####
############################

# Dictionary defining the properties of the model domain
domain = dict(xmin=0,
              xmax=4,
              ymin=0,
              ymax=4,
              por=0.35,
              por_tolerance=0.05,
              height=model["mindist"]/np.sqrt(8))

####################################
####    MODELLING PARAMETERS    ####
####################################

# Which steps to perform
generate_domain = True
pre_process = True
simulations = True
post_process = True

# Whether or not to perform mesh refinement around grains during pre-processing
snappy_refinement = False

# Margin around the edges of the domain that will be ignored during post-processing
post_processing_margin = (domain["xmax"] - domain["xmin"]) * 0.1

# Number of allowed failed cases before the program stops
n_allowed_fails = 1

# Number of simulations to do during Monte Carlo
n_simulations = 1
# Number of cores to use for each simulation
cores_per_sim = 1
# Whether or not to run on cluster (non-cluster simulations not yet implemented)
cluster = False
# Username of account of cluster from which this script is being run
cluster_user = "3979202"
# Number of simulations to run in parallel at one time (value higher than 1 only supported for cluster)
n_parallel_sims = 10
# Amount tasks to be invoked per computing node
tasks_per_node = 1
# Restrict node selection to nodes with at least the specified number of threads per core.
threads_per_core = 2
# Which queue to run simulations on
partition = "allq"

# List of modules to load when running bash script on cluster
modules = ["opt/all",
           "gcc/6.4.0",
           "openmpi/gcc-6.4.0/3.1.2",
           "openFoam/6"]
# List of scripts to source when running bash script on cluster
scripts = ["/trinity/opt/apps/software/openFoam/version6/OpenFOAM-6/etc/bashrc"]

# Name of the base directory to perform the simulations in
base_dir = "../../Simulations"
# Name of this batch of simulations
run_name = "Cyclic_fixed_mesh_test"
# Directory to copy the base OpenFOAM case from
basecase_dir = "../baseCase_cyclic"

this_dir = os.getcwd()

# Get real paths
base_dir = os.path.realpath(base_dir)
basecase_dir = os.path.realpath(basecase_dir)

# Check if baseDir path already exists, if not, create it
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
os.chdir(base_dir)

### DOMAIN GENERATION ###

if generate_domain:
    print("Starting domain generation")
    if not os.path.isdir("stl"):
        os.mkdir("stl")
    os.chdir("stl")

    for i in range(n_simulations):
        if not os.path.isdir("{0}_{1}".format(run_name, i)):
            os.mkdir("{0}_{1}".format(run_name, i))
        os.chdir("{0}_{1}".format(run_name, i))

        # Generate and save model as .stl file
        randomCircles.create_model(model, domain, stl_filename="stl")
        os.chdir("..")
    os.chdir("..")

### PRE-PROCESSING ###

if pre_process:
    print("Starting pre-processing")
    if not os.path.isdir("cases"):
        os.mkdir("cases")

    if not os.path.isdir(basecase_dir):
        print("ERROR: given OpenFOAM base case directory '{0}' does not exist".format(basecase_dir))
        quit()

    if not os.path.isdir("baseCase"):
        os.mkdir("baseCase")
    subprocess.call("cp -rf {0}{1}* baseCase{1}".format(basecase_dir, os.sep), shell=True)

    # Get absolute paths to stl, baseCase & cases directories
    stl_dir = os.path.realpath("stl")
    basecase_dir = os.path.realpath("baseCase")
    cases_dir = os.path.realpath("cases")

    for i in range(n_simulations):
        # Get reference to directory of this case, this case's stl directory and to this case's .stl file
        case_dir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i))
        case_stl_dir = os.path.realpath("{0}{1}{2}_{3}".format(stl_dir, os.sep, run_name, i))
        case_stl_file = os.path.realpath("{0}{1}stl.stl".format(case_stl_dir, os.sep, i))

        # Create case from baseCase if it does not exist yet
        if not os.path.isdir(case_dir):
            subprocess.call(["cp", "-rf", basecase_dir, case_dir])

        # Go into case directory
        os.chdir(case_dir)

        # Copy .stl file to the triSurface folder of the case
        subprocess.call(["cp", case_stl_file, "{0}{1}constant{1}triSurface".format(case_dir, os.sep)])

        # Get location in mesh from stl directory
        lim_file = open("{0}{1}locationInMesh.dat".format(case_stl_dir, os.sep))
        location_in_mesh = map(float, lim_file.readline().split())
        lim_file.close()

        # Update the blockMeshDict, snappyHexMeshDict and decomposeParDict of the case according to given parameters
        preP.update_blockMeshDict("system", domain, mindist=model["mindist"])
        preP.update_snappyHexMeshDict("system", "stl", domain["height"], model["mindist"], location_in_mesh, refinement=snappy_refinement, castellated_mesh=True, snap=True if not snappy_refinement else False)
        preP.update_decomposeParDict("system", cores_per_sim)
        preP.update_extrudeMeshDict("system", domain["height"])
        preP.create_pre_processing_script("{0}_{1}".format(run_name, i), cores_per_sim, tasks_per_node, threads_per_core, partition, modules, scripts, refinement=snappy_refinement)

        # Go back to base directory
        os.chdir(base_dir)

    # Start meshing
    waiting_cases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i)) for i in range(n_simulations)]
    active_cases = []
    finished_cases = []

    if cluster:
        while len(finished_cases) < n_simulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waiting_cases and len(active_cases) < n_parallel_sims:
                new_case_dir = waiting_cases.pop(0)
                active_cases.append(new_case_dir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(new_case_dir)
                subprocess.call(["sbatch", "{0}{1}preprocessing{2}.sh".format(new_case_dir, os.sep, "_0" if snappy_refinement else ""), "&"])
                os.chdir(base_dir)

            print("Cases running/in queue: {0}".format(active_cases))

            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            end_cases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32,JobID").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that started the process
            queue = dict([[i[:32].strip(), [i[32:64].strip(), i[64:].strip()]] for i in queue])
            for i in range(len(active_cases)):
                # Get references to case directory and case name
                case_dir = active_cases[i]
                case_name = case_dir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_pre".format(case_name) in queue:
                    if snappy_refinement and not os.path.isfile("{0}{1}snappyHexMesh_1.log".format(case_dir, os.sep)):
                        # Start second part of meshing if refinement is on and the second snappyHexMesh log has not been created yet
                        os.chdir(case_dir)

                        # Get the stl directory corresponding to this case
                        case_num = case_dir.split("_")[-1]
                        case_stl_dir = os.path.realpath("{0}{1}{2}_{3}".format(stl_dir, os.sep, run_name, case_num))

                        # Get location in mesh from stl directory
                        lim_file = open("{0}{1}locationInMesh.dat".format(case_stl_dir, os.sep))
                        location_in_mesh = map(float, lim_file.readline().split())
                        lim_file.close()

                        preP.update_snappyHexMeshDict("system", "stl", domain["height"], model["mindist"], location_in_mesh, refinement=snappy_refinement, castellated_mesh=False, snap=True)
                        subprocess.call(["sbatch", "{0}{1}preprocessing_1.sh".format(case_dir, os.sep), "&"])
                        os.chdir(base_dir)
                    else:
                        # Insert the index of finished case at the beginning of the endCases list, so it will be ordered from high to low index
                        end_cases.insert(0, i)

            for i in end_cases:
                case = active_cases.pop(i)
                finished_cases.append(case)
                print("Finished cases: {0}".format(finished_cases))

        print("All cases finished")
    else:
        while waiting_cases:
            case_dir = waiting_cases.pop(0)
            os.chdir(case_dir)
            subprocess.call(["chmod", "+x",
                             "{0}{1}preprocessing{2}.sh".format(case_dir, os.sep, "_0" if snappy_refinement else "")])
            subprocess.call(["./preprocessing{0}.sh".format("_0" if snappy_refinement else "")])
            if snappy_refinement:
                # Get the stl directory corresponding to this case
                case_num = case_dir.split("_")[-1]
                case_stl_dir = os.path.realpath("{0}{1}{2}_{3}".format(stl_dir, os.sep, run_name, case_num))

                # Get location in mesh from stl directory
                lim_file = open("{0}{1}locationInMesh.dat".format(case_stl_dir, os.sep))
                location_in_mesh = map(float, lim_file.readline().split())
                lim_file.close()

                preP.update_snappyHexMeshDict("system", "stl", domain["height"], model["mindist"], location_in_mesh, refinement=snappy_refinement, castellated_mesh=False, snap=True)
                subprocess.call(["chmod", "+x", "{0}{1}preprocessing_1.sh".format(case_dir, os.sep)])
                subprocess.call(["./preprocessing_1.sh"])
            finished_cases.append(case_dir)
            os.chdir(base_dir)

### SIMULATION ###

os.chdir(base_dir)
if simulations:
    print("Starting simulations")
    for i in range(n_simulations):
        case_dir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i))
        if not os.path.isdir(case_dir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(run_name, i))
            continue
        os.chdir(case_dir)
        preP.create_simulation_script("{0}_{1}".format(run_name, i), cores_per_sim, tasks_per_node, threads_per_core, partition, modules, scripts)
        os.chdir(base_dir)

    n_sim_fails = 0

    waiting_cases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i)) for i in range(n_simulations)]
    active_cases = []
    finished_cases = []

    if cluster:
        while len(finished_cases) < n_simulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waiting_cases and len(active_cases) < n_parallel_sims:
                new_case_dir = waiting_cases.pop(0)
                active_cases.append(new_case_dir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(new_case_dir)
                subprocess.call(["sbatch", "{0}{1}runSimulations.sh".format(new_case_dir, os.sep), "&"])
                os.chdir(base_dir)
                print("Cases running/in queue: {0}".format(active_cases))
            
            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            end_cases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that
            # started the process
            queue = dict([[i[:32].strip(), i[32:].strip()] for i in queue])
            for i in range(len(active_cases)):
                # Get references to case directory and case name
                case_dir = active_cases[i]
                case_name = case_dir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_sim".format(case_name) in queue:
                    end_cases.insert(0, i)

            for i in end_cases:
                case = active_cases.pop(i)
                if preP.check_log("{0}{1}simpleFoam.log".format(case, os.sep)):
                    finished_cases.append(case)
                    print("Finished cases: {0}".format(finished_cases))
                else:
                    n_sim_fails += 1
                    case_name = case.split(os.sep)[-1]
                    print("Case '{0}' failed to run properly. Current total number of fails: {1}/{2}".format(case_name, n_sim_fails, n_allowed_fails))
                    if n_sim_fails >= n_allowed_fails:
                        print("Maximum amount of failed cases reached ({0}), quitting...".format(n_allowed_fails))
                        # Add some code here to stop all other simulations?
                        quit()
                    else:
                        print("Restarting case '{0}'".format(case_name))
                        waiting_cases.append(case)
    else:
        while waiting_cases:
            case_dir = waiting_cases.pop(0)
            os.chdir(case_dir)
            subprocess.call(["chmod", "+x", "runSimulations.sh"])
            subprocess.call(["./runSimulations.sh"])
            if preP.check_log("{0}{1}simpleFoam.log".format(case_dir, os.sep)):
                finished_cases.append(case_dir)
            else:
                n_sim_fails += 1
                case_name = case_dir.split(os.sep)[-1]
                print("Case '{0}' failed to run properly. Current total number of fails: {1}/{2}".format(
                    case_name, n_sim_fails, n_allowed_fails))
                if n_sim_fails >= n_allowed_fails:
                    print("Maximum amount of failed cases reached ({0}), quitting...".format(n_allowed_fails))
                    quit()
                else:
                    print("Restarting case '{0}'".format(case_name))
                    waiting_cases.append(case_dir)
            os.chdir(base_dir)
    print("All cases finished")

os.chdir(base_dir)
if post_process:
    print("Starting post-processing")
    for i in range(n_simulations):
        case_dir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i))
        if not os.path.isdir(case_dir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(run_name, i))
            continue
        os.chdir(case_dir)

        # Get filename of .vtk file of final iteration
        if not os.path.isdir("VTK"):
            print("WARNING: no directory 'VTK' in case directory '{0}_{1}', skipping this case".format(run_name, i))
        os.chdir("VTK")
        vtk_files = []
        for item in os.listdir("."):
            if item.lower().endswith(".vtk") and os.path.isfile(item):
                vtk_files.append(item)
        if not vtk_files:
            print("WARNING: no .vtk files in 'VTK' directory of case '{0}_{1}', skipping this case".format(run_name, i))
        vtk_files.sort()
        vtk_file = vtk_files[-1]
        os.chdir(case_dir)

        # Create script for running post-processing on cluster
        postP.create_post_processing_script(os.path.realpath("."), "{0}_{1}".format(run_name, i), "{0}{1}postProcessing.py".format(this_dir, os.sep), "VTK{0}{1}".format(os.sep, vtk_file), 10 ** -6, 1000, margin=post_processing_margin)

        os.chdir(base_dir)

    waiting_cases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i)) for i in range(n_simulations)]
    active_cases = []
    finished_cases = []

    if cluster:
        while len(finished_cases) < n_simulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waiting_cases and len(active_cases) < n_parallel_sims:
                new_case_dir = waiting_cases.pop(0)
                active_cases.append(new_case_dir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(new_case_dir)
                subprocess.call(["sbatch", "{0}{1}postprocessing.sh".format(new_case_dir, os.sep), "&"])
                os.chdir(base_dir)

            print("Cases running/in queue: {0}".format(active_cases))
            
            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            end_cases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that
            # started the process
            queue = dict([[i[:32].strip(), i[32:].strip()] for i in queue])
            for i in range(len(active_cases)):
                # Get references to case directory and case name
                case_dir = active_cases[i]
                case_name = case_dir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_post".format(case_name) in queue:
                    end_cases.insert(0, i)

            for i in end_cases:
                case = active_cases.pop(i)
                finished_cases.append(case)
                print("Finished cases: {0}".format(finished_cases))
        print("All cases finished")
    else:
        while waiting_cases:
            case_dir = waiting_cases.pop(0)
            os.chdir(case_dir)
            subprocess.call(["chmod", "+x", "postprocessing.sh"])
            subprocess.call(["./postprocessing.sh"])
            finished_cases.append(case_dir)
            os.chdir(base_dir)

    if not os.path.isfile("results.dat"):
        out_file = open("results.dat", "w")
        out_file.write("Simulation\tPorosity_[-]\tPermeability_[m^2]\n")
    else:
        out_file = open("results.dat", "a+")

    for i in range(n_simulations):
        case_dir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i))
        if not os.path.isdir(case_dir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(run_name, i))
            continue
        os.chdir(case_dir)

        case_result = open("out.dat", "r")
        por, k = map(float, case_result.readline().strip().split(","))
        case_result.close()

        out_file.write("{0}\t{1}\t{2}\n".format("{0}_{1}".format(run_name, i), por, k))

        os.chdir(base_dir)

    out_file.close()
