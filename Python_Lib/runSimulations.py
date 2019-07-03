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
model = dict(distributionType="truncLogNormal",
            rmin=0.05,
            rmax=0.8,
            rmean=0.35,
            rstd=0.25,
            mindist=0.025)

############################
####    DOMAIN SIZE     ####
############################

# Dictionary defining the properties of the model domain
domain = dict(xmin=0,
            xmax=4,
            ymin=0,
            ymax=4,
            por=0.35,
            porTolerance=0.05,
            height=model["mindist"]/np.sqrt(8))

####################################
####    MODELLING PARAMETERS    ####
####################################

# Which steps to perform
generateDomain = False
preProcess = True
simulations = True
postProcess = True

# Whether or not to perform mesh refinement around grains during pre-processing
snappyRefinement = False

# Margin around the edges of the domain that will be ignored during post-processing
postProcessingMargin = (domain["xmax"] - domain["xmin"]) * 0.1

# Number of allowed failed cases before the program stops
nAllowedFails = 10

# Number of simulations to do during Monte Carlo
nSimulations = 1
# Number of cores to use for each simulation
coresPerSim = 1
# Whether or not to run on cluster (non-clustervsimulations not yet implemented)
cluster = False
# Username of account of cluster from which this script is being run
clusterUser = "3979202"
# Number of simulations to run in parallel at one time (value higher than 1 only supported for cluster)
nParallelSims = 10
# Amount tasks to be invoked per computing node
tasksPerNode = 1
# Restrict node selection to nodes with at least the specified number of threads per core.
threadsPerCore = 2
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
baseDir = "../Simulations"
# Name of this batch of simulations
runName = "Cyclic"
# Directory to copy the base OpenFOAM case from
baseCaseDir = "../baseCase_cyclic"

thisDir = os.getcwd()

# Get real paths
baseDir = os.path.realpath(baseDir)
baseCaseDir = os.path.realpath(baseCaseDir)

# Check if baseDir path already exists, if not, create it
if not os.path.isdir(baseDir):
    os.mkdir(baseDir)
os.chdir(baseDir)

### DOMAIN GENERATION ###

if generateDomain:
    print("Starting domain generation")
    if not os.path.isdir("stl"):
        os.mkdir("stl")
    os.chdir("stl")

    for i in range(nSimulations):
        if not os.path.isdir("{0}_{1}".format(runName, i)):
            os.mkdir("{0}_{1}".format(runName, i))
        os.chdir("{0}_{1}".format(runName, i))

        # Generate and save model as .stl file
        randomCircles.createModel(model, domain, stlFilename="stl")
        os.chdir("..")
    os.chdir("..")

### PRE-PROCESSING ###

if preProcess:
    print("Starting pre-processing")
    if not os.path.isdir("cases"):
        os.mkdir("cases")

    if not os.path.isdir(baseCaseDir):
        print("ERROR: given OpenFOAM base case directory '{0}' does not exist".format(baseCaseDir))
        quit()

    if not os.path.isdir("baseCase"):
        os.mkdir("baseCase")
    subprocess.call(["cp", "-rf", "{0}{1}* baseCase{1}".format(baseCaseDir, os.sep)])

    # Get absolute paths to stl, baseCase & cases directories
    stlDir = os.path.realpath("stl")
    baseCaseDir = os.path.realpath("baseCase")  # Old baseCadeDir no longer needed, since it has been copied into the baseDir
    casesDir = os.path.realpath("cases")

    for i in range(nSimulations):
        # Get reference to directory of this case, this case's stl directory and to this case's .stl file
        caseDir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i))
        caseStlDir = os.path.realpath("{0}{1}{2}_{3}".format(stlDir, os.sep, runName, i))
        caseStlFile = os.path.realpath("{0}{1}stl.stl".format(caseStlDir, os.sep, i))

        # Create case from baseCase if it does not exist yet
        if not os.path.isdir(caseDir):
            subprocess.call(["cp", "-rf", baseCaseDir, caseDir])

        # Go into case directory
        os.chdir(caseDir)

        # Copy .stl file to the triSurface folder of the case
        subprocess.call(["cp", caseStlFile, "{0}{1}constant{1}triSurface".format(caseDir, os.sep)])

        # Get location in mesh from stl directory
        LIM_file = open("{0}{1}locationInMesh.dat".format(caseStlDir, os.sep))
        locationInMesh = map(float, LIM_file.readline().split())
        LIM_file.close()

        # Update the blockMeshDict, snappyHexMeshDict and decomposeParDict of the case according to given parameters
        preP.updateBlockMeshDict("system", domain["xmin"], domain["xmax"], domain["ymin"], domain["ymax"], domain["height"], mindist=model["mindist"])
        preP.updateSnappyHexMeshDict("system", "stl", locationInMesh, refinement=snappyRefinement, castellatedMesh=True, snap=True if not snappyRefinement else False)
        preP.updateDecomposeParDict("system", coresPerSim)
        preP.updateExtrudeMeshDict("system", height)
        preP.createPreProcessingScript("{0}_{1}".format(runName, i), coresPerSim, tasksPerNode, threadsPerCore, partition, modules, scripts, refinement=snappyRefinement)

        # Go back to base directory
        os.chdir(baseDir)

    # Start meshing
    waitingCases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i)) for i in range(nSimulations)]
    activeCases = []
    finishedCases = []

    if cluster:
        while len(finishedCases) < nSimulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waitingCases and len(activeCases) < nParallelSims:
                newCaseDir = waitingCases.pop(0)
                activeCases.append(newCaseDir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(newCaseDir)
                subprocess.call(["sbatch", "{0}{1}preprocessing{2}.sh".format(newCaseDir, os.sep, "_0" if snappyRefinement else ""), "&"])
                os.chdir(baseDir)

            print("Cases running/in queue: {0}".format(activeCases))

            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            endCases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32,JobID").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that started the process
            queue = dict([[i[:32].strip(), [i[32:64].strip(), i[64:].strip()]] for i in queue])
            for i in range(len(activeCases)):
                # Get references to case directory and case name
                caseDir = activeCases[i]
                caseName = caseDir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_pre".format(caseName) in queue:
                    if snappyRefinement and not os.path.isfile("{0}{1}snappyHexMesh_1.log".format(caseDir, os.sep)):
                        # Start second part of meshing if refinement is on and the second snappyHexMesh log has not been created yet
                        os.chdir(newCaseDir)
                        updateSnappyHexMeshDict("system", "stl", refinement=snappyRefinement, castellatedMesh=False, snap=True)
                        subprocess.call(["sbatch", "{0}{1}preprocessing_1.sh".format(caseDir, os.sep), "&"])
                        os.chdir(baseDir)
                    else:
                        # Insert the index of finished case at the beginnning of the endCases list, so it will be ordered from high to low index
                        endCases.insert(0, i)

            for i in endCases:
                case = activeCases.pop(i)
                finishedCases.append(case)
                print("Finished cases: {0}".format(finishedCases))

        print("All cases finished")
    else:
        while waitingCases:
            caseDir = waitingCases.pop(0)
            os.chdir(caseDir)
            subprocess.call(["chmod", "+x", "{0}{1}preprocessing{2}.sh".format(caseDir, os.sep, "_0" if snappyRefinement else "")])
            subprocess.call(["./preprocessing{0}.sh".format("_0" if snappyRefinement else "")])
            if snappyRefinement:
                preP.updateSnappyHexMeshDict("system", "stl", refinement=snappyRefinement, castellatedMesh=False, snap=True)
                subprocess.call(["chmod", "+x", "{0}{1}preprocessing_1.sh".format(caseDir, os.sep)])
                subprocess.call(["./preprocessing_1.sh"])
            finishedCases.append(caseDir)
            os.chdir(baseDir)

### SIMULATION ###

os.chdir(baseDir)
if simulations:
    print("Starting simulations")
    for i in range(nSimulations):
        caseDir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i))
        if not os.path.isdir(caseDir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(runName, i))
            continue
        os.chdir(caseDir)
        preP.createSimulationScript("{0}_{1}".format(runName, i), coresPerSim, tasksPerNode, threadsPerCore, partition, modules, scripts)
        os.chdir(baseDir)

    nSimFails = 0

    waitingCases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i)) for i in range(nSimulations)]
    activeCases = []
    finishedCases = []

    if cluster:
        while len(finishedCases) < nSimulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waitingCases and len(activeCases) < nParallelSims:
                newCaseDir = waitingCases.pop(0)
                activeCases.append(newCaseDir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(newCaseDir)
                subprocess.call(["sbatch", "{0}{1}runSimulations.sh".format(newCaseDir, os.sep), "&"])
                os.chdir(baseDir)
                print("Cases running/in queue: {0}".format(activeCases))
            
            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            endCases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that started the process
            queue = dict([[i[:32].strip(), i[32:].strip()] for i in queue])
            for i in range(len(activeCases)):
                # Get references to case directory and case name
                caseDir = activeCases[i]
                caseName = caseDir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_sim".format(caseName) in queue:
                    endCases.insert(0, i)

            for i in endCases:
                case = activeCases.pop(i)
                if checkLog("{0}{1}simpleFoam.log".format(case)):
                    finishedCases.append(case)
                    print("Finished cases: {0}".format(finishedCases))
                else:
                    nSimFails += 1
                    caseName = case.split(os.sep)[-1]
                    print("Case '{0}' failed to run properly. Current total number of fails: {1}/{2}".format(caseName, nSimFails, nAllowedFails))
                    if nSimFails >= nAllowedFails:
                        print("Maximum amount of failed cases reached ({0}), quitting...".format(nAllowedFails))
                        # Add some code here to stop all other simulations?
                        quit()
                    else:
                        print("Restarting case '{0}'".format(caseName))
                        waitingCases.append(case)
    else:
        while waitingCases:
            caseDir = waitingCases.pop(0)
            os.chdir(caseDir)
            subprocess.call(["chmod", "+x", "runSimulations.sh"])
            subprocess.call(["./runSimulations.sh"])
            if checkLog("{0}{1}simpleFoam.log".format(caseDir, os.sep))
                finishedCases.append(caseDir)
            else:
                nSimFails += 1
                caseName = caseDir.split(os.sep)[-1]
                print("Case '{0}' failed to run properly. Current total number of fails: {1}/{2}".format(caseName, nSimFails, nAllowedFails))
                if nSimFails >= nAllowedFails:
                    print("Maximum amount of failed cases reached ({0}), quitting...".format(nAllowedFails))
                    quit()
                else:
                    print("Restarting case '{0}'".format(caseName))
                    waitingCases.append(caseDir)
            os.chdir(baseDir)
    print("All cases finished")

os.chdir(baseDir)
if postProcess:
    print("Starting post-processing")
    for i in range(nSimulations):
        caseDir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i))
        if not os.path.isdir(caseDir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(runName, i))
            continue
        os.chdir(caseDir)

        # Get filename of .vtk file of final iteration
        if not os.path.isdir("VTK"):
            print("WARNING: no directory 'VTK' in case directory '{0}_{1}', skipping this case".format(runName, i))
        os.chdir("VTK")
        vtkFiles = []
        for item in os.listdir():
            if item.lower().endswith(".vtk") and os.path.isfile(item):
                vtkFiles.append(item)
        if not vtkFiles:
            print("WARNING: no .vtk files in 'VTK' directory of case '{0}_{1}', skipping this case".format(runName, i))
        vtkFiles.sort()
        vtkFile = vtkFiles[-1]
        os.chdir(caseDir)

        # Create script for running post-processing on cluster
        postP.createPostProcessingScript(os.path.realpath("."), "{0}_{1}".format(runName, i), "{0}{1}postProcessing.py".format(thisDir, os.sep),  "VTK{0}{1}".format(os.sep, vtkFile), 10**-6, 1000, margin=postProcessingMargin)

        os.chdir(baseDir)

    waitingCases = [os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i)) for i in range(nSimulations)]
    activeCases = []
    finishedCases = []  

    if cluster:
        while len(finishedCases) < nSimulations:
            # Start new cases if there are still waiting cases and there are free slots
            while waitingCases and len(activeCases) < nParallelSims:
                newCaseDir = waitingCases.pop(0)
                activeCases.append(newCaseDir)
                # Put new case in queue (note the '&' which prevents the command from blocking the program)
                os.chdir(newCaseDir)
                subprocess.call(["sbatch", "{0}{1}postprocessing.sh".format(newCaseDir, os.sep), "&"])
                os.chdir(baseDir)

            print("Cases running/in queue: {0}".format(activeCases))
            
            # Give the cluster some time to put the scripts into the queue
            time.sleep(5)

            # Initialize list which will hold the indices of cases which are finished
            endCases = []
            # Get list of processes that are in the queue or running, together with the username
            queue = os.popen("squeue -h --Format=name:32,username:32").read()
            queue = queue.split("\n")[:-1]
            # Create dictionary with keys being the names of processes in the queue and values the usernames that started the process
            queue = dict([[i[:32].strip(), i[32:].strip()] for i in queue])
            for i in range(len(activeCases)):
                # Get references to case directory and case name
                caseDir = activeCases[i]
                caseName = caseDir.split(os.sep)[-1]
                # Check if case is still in the queue (currently username is not taken into consideration)
                if not "{0}_post".format(caseName) in queue:
                    endCases.insert(0, i)

            for i in endCases:
                case = activeCases.pop(i)
                finishedCases.append(case)
                print("Finished cases: {0}".format(finishedCases))
        print("All cases finished")
    else:
        while waitingCases:
            caseDir = waitingCases.pop(0)
            os.chdir(caseDir)
            subprocess.call(["chmod", "+x", "postprocessing.sh"])
            subprocess.call(["./postprocessing.sh"])
            finishedCases.append(caseDir)
            os.chdir(baseDir)

    if not os.path.isfile("results.dat"):
        outFile = open("results.dat", "w")
        outFile.write("Simulation\tPorosity_[-]\tPermeability_[m^2]\n")
    else:
        outFile = open("results.dat", "a+")

    for i in range(nSimulations):
        caseDir = os.path.realpath("cases{0}{1}_{2}".format(os.sep, runName, i))
        if not os.path.isdir(caseDir):
            print("WARNING: case directory '{0}_{1}' does not exist, skipping this case".format(runName, i))
            continue
        os.chdir(caseDir)

        caseResult = open("out.dat", "r")
        por, k = map(float, caseResult.readline().strip().split(","))
        caseResult.close()

        outFile.write("{0}\t{1}\t{2}\n".format("{0}_{1}".format(runName, i), por, k))

    outFile.close()

def checkLog(logfile):
    """Checks OpenFOAM log file to see if an OpenFOAM process ended properly or aborted due to an error.
Returns True if log ended properly, else returns False.

PARAMETERS
----------
logfile : str
    Path to the log file to be checked.

RETURNS
-------
status : bool
    True or False value depending on whether or not the OpenFOAM process ended properly, respectively."""
    
    # Get the last word from the log file using the 'tail' command
    lastWord = os.popen("tail {0}".format(logFile)).read().split()[-1]

    # If log file ends with the word 'End', we know that the process ended properly, otherwise something went wrong
    if lastWord == "End":
        status = True
    else:
        status = False

    return status