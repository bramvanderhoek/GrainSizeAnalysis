#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:35:54 2019

@author: zech0001
"""
import numpy as np
import os
import subprocess
import time

### Local dependencies
import randomCircles
import preProcessing as preP
import postProcessing as postP
import writeScripts as writeS

###############################################################################
###############################################################################
###############################################################################


class Ensemble:
    def __init__(
        self,
        grain_model,
        domain,
        dirs,
        n_realizations=1,
        cluster=False,
        bc="periodic",
        **cluster_kwargs
    ):

        self.grain_model = grain_model
        self.domain = domain
        self.dirs = dirs
        self.n_realizations = n_realizations
        self.bc = bc      
        self.cluster=cluster
        
        self.setup_dir_structure()
        self.set_unit()
        self.set_seed()
        self.write_settings()

        self.set_cluster_kwargs()
        self.cluster_kwargs.update(cluster_kwargs)           

    def setup_dir_structure(self):
        if "dir_root" in self.dirs:
            self.dir_root = self.dirs["dir_root"]
        else:
            # self.dir_root=os.getcwd()+'/'
            self.dir_root = os.path.realpath("../") + "/"

        if "name_run" in self.dirs:
            self.name_run = self.dirs["name_run"]
        else:
            self.name_run = "field_data"

        # Check if ensemble directory already exists, if not, create it
        self.dir_sim = "{}{}/".format(self.dir_root,self.name_run)
        if not os.path.isdir(self.dir_sim):
            os.mkdir(self.dir_sim)

        # Check if stl directory (containing geometries) already exists, if not, create it
        self.dir_stl = "{}{}/".format(self.dir_sim,"stl")
        if not os.path.isdir(self.dir_stl):
            os.mkdir(self.dir_stl)

        self.dir_stl_files = "{}{}{}/".format(self.dir_stl, self.name_run,"_{}")
        #self.dir_stl_files = self.dir_stl + "/" + self.name_run + "_{}"
        self.name_stl_file = "stl"

        # Check if cases directory (containing simulations) already exists, if not, create it
        self.dir_cases = "{}{}/".format(self.dir_sim,"cases")
        if not os.path.isdir(self.dir_cases):
            os.mkdir(self.dir_cases)
        #self.dir_case_sim = "{}{}".format(self.dir_cases,self.name_run) + "_{}"
        self.dir_case_sim = "{}{}{}/".format(self.dir_cases, self.name_run,"_{}")


        # Setup base case (containing openfoam standard simulation settings)
        if self.bc in ["periodic", "cyclic"]:
            dir_base = self.dirs["basecase_cyclic"]
        else:
            dir_base = self.dirs["basecase_symmetry"]

        self.dir_basecase = "{}{}".format(self.dir_sim,"baseCase")
        if not os.path.isdir(dir_base):
            raise ValueError(
                "OpenFOAM base case directory '{0}' does not exist".format(
                    self.dir_basecase
                )
            )
        elif not os.path.isdir(self.dir_basecase):
            subprocess.call(
                "cp -rf {0} {1}".format(dir_base, self.dir_basecase), shell=True
            )

        # Set up python working directory to return from local command line runs
        self.python_wd = os.getcwd()

    def set_unit(self,unit=False):

        if 'unit' in self.domain:
            unit=self.domain['unit']
        
        if unit=='m':
            self.unit_factor=1.
        elif unit=='cm':
            self.unit_factor=0.01
        elif unit=='mm':
            self.unit_factor=0.001
        else:
            print("Warning: Unit not properly defined, using value of 1.")
            self.unit_factor=1.

    def set_cluster_kwargs(self):

        self.cluster_kwargs = dict(
            cores_per_sim=1,
            n_parallel_sims=12,  # Number of simulations to run in parallel at one time (value higher than 1 only supported for cluster)
            tasks_per_node=1,  # Amount tasks to be invoked per computing node
            threads_per_core=2,  # Restrict node selection to nodes with at least the specified number of threads per core.
            partition="allq",  # Which queue to run simulations on
            modules = ["opt/all",           
                       "gcc/6.4.0",
                       "openmpi/gcc-6.4.0/3.1.2",
                       "openFoam/6"],
            scripts=[ # List of scripts to source when running bash script on cluster
                "/trinity/opt/apps/software/openFoam/version6/OpenFOAM-6/etc/bashrc"],  
            )

    def set_seed(self,seed=False):

        # Since the seed entry of the model dictionary will be updated per simulation, keep track of the provided base seed
        if "seed" in self.grain_model:
            self.seed = self.grain_model["seed"]
        else:
            self.seed = seed

        if self.seed is not False:
            if self.seed is True:
                self.base_seed = self.seed = np.random.randint(0, 10 ** 9)
            else:
                self.base_seed = self.seed

        return self.base_seed

    def write_settings(self, file_settings="settings.txt"):

        if "file_settings" in self.dirs:
            file_settings = self.dirs["file_settings"]
        else:
            file_settings = "{}/{}".format(self.dir_sim,file_settings)

        if not os.path.isfile(file_settings):
            settings_file = open(file_settings, "w")
            header_items = [
                "run_name",
                "date/time",
                "n-realizations",
                "distribution",
                "min_dist",
                "base_seed",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "por_min",
                "por_max",
                "cell size",
            ]
            settings_file.write(
                "{0:<31} {1:<20} {2:<20} {3:<70} {4:<7} {5:<15} {6:<7} {7:<7} {8:<7} {9:<7} {10:<7} {11:<7} {12:<7}\n".format(
                    *header_items
                )
            )
        else:
            settings_file = open(file_settings, "a+")

        localtime = time.localtime()
        date_time = "{0}/{1}/{2} {3:02}:{4:02}:{5:02}".format(
            localtime[2],
            localtime[1],
            localtime[0],
            localtime[3],
            localtime[4],
            localtime[5],
        )

        dist = grain_model["model_data"]
        if self.grain_model["model_type"] == "trunc_lognormal":
            dist_string = "truncLogNormal: rmin={0}, rmax={1}, rmean={2}, rstd={3}".format(
                dist["rmin"], dist["rmax"], dist["rmean"], dist["rstd"]
            )
        elif self.grain_model["model_type"] == "data_distribution":
            dist_string = "(data: data_points={0}, c_freq={1})".format(
                dist["data_points"], dist["c_freq"]
            )

        setting_items = [
            self.name_run[:31],
            date_time,
            self.n_realizations,
            dist_string,
            self.grain_model["mindist"],
            self.seed,
            self.domain["xmin"],
            self.domain["xmax"],
            self.domain["ymin"],
            self.domain["ymax"],
            self.domain["por_min"],
            self.domain["por_max"],
            self.domain["cell_size"],
        ]
        settings_file.write(
            "{0:<31} {1:<20} {2:<20} {3:<70} {4:<7} {5:<15} {6:<7} {7:<7} {8:<7} {9:<7} {10:<7} {11:<7} {12:<7}\n".format(
                *setting_items
            )
        )
        settings_file.close()

    def generate_geometries(self,overwrite=False):

        """

        Optional Arguments
        ------------------
        overwrite       - Whether or not to overwrite stl files if they already exist

        """

        print("\n### STARTING DOMAIN GENERATION ###\n\n")

        for i in range(self.n_realizations):

            if not os.path.isdir(self.dir_stl_files.format(i)):
                os.mkdir(self.dir_stl_files.format(i))
            file_stl = "{}{}{}".format(self.dir_stl_files.format(i), self.name_stl_file, ".stl")

            if writeS.check_file(file_stl,i=i,overwrite=overwrite,process="Domain creation",log=False):
                continue

            if self.seed:  # If seed is in use, update it for this specific simulation
                self.grain_model["seed"] = self.base_seed + i

            por = randomCircles.create_model(
                self.grain_model,
                self.domain,
                stl_filename=self.name_stl_file,
                path=self.dir_stl_files.format(i),
            )  # , plotting=True)

            while not domain["por_min"] <= por <= domain["por_max"]:
                print("Porosity not within specified range, recreating geometry")
                if self.grain_model["seed"] is not False:
                    print(
                        "WARNING: provided seed was unable to generate \
                        correct geometry, random seed will be used for \
                        case {0}".format(i))
                    self.grain_model["seed"] = np.random.randint(0, 10 ** 9)+i
                por = randomCircles.create_model(
                    self.grain_model,
                    self.domain,
                    stl_filename=self.name_stl_file,
                    path=self.dir_stl_files.format(i),
                )

        print("\n# Domains for {:d} realizations set-up #".format(self.n_realizations))

    def pre_process(
        self,
        overwrite=False,
        snapping=True,
        snappy_refinement=False,
        **cluster_kwargs
    ):  

        """
        Optional Arguments
        ------------------
        snapping            - Whether or not to perform mesh snapping to grain surfaces during pre-processing                          
        snappy_refinement   - Whether or not to perform mesh refinement around grains during pre-processing
             - 
        
        cluster_args        - cluster arguments
            tasks_per_node      - Amount tasks to be invoked per computing node
            threads_per_core    - Restrict node selection to nodes with at least the specified number of threads per core.
            partition           - Which queue to run simulations on
            modules             - List of modules to load when running bash script on cluster
            scripts             - List of scripts to source when running bash script on cluster
        """

        print("\n### STARTING PRE-PROCESSING ###\n\n")
        self.cluster_kwargs.update(cluster_kwargs)           

        if self.bc in ["periodic", "cyclic"] and self.cluster_kwargs['cores_per_sim'] > 1:
            print(
                "### WARNING:  using multiple cores for a case with cyclic BC is error prone"
            )

        for i in range(self.n_realizations):

            # Get reference to directory of this case, this case's stl directory and to this case's .stl file
            case_dir = self.dir_case_sim.format(i)
            dir_stl = self.dir_stl_files.format(i) 
            stl_file = "{}{}{}".format(dir_stl, self.name_stl_file,".stl")

            if writeS.check_file("{0}/snappyHexMesh_0.log".format(case_dir),i=i,
                                 process='Preprocessing',overwrite=overwrite,log=True):
                continue

            # Create case from baseCase if it does not exist yet, or if existing directory is not a valid OpenFOAM directory
            openfoam_dirs = [
                "{}{}".format(case_dir, folder)
                for folder in ["0", "constant", "system"]
            ]
            openfoam_case = np.array(
                [os.path.isdir(folder) for folder in openfoam_dirs]).all()
            if not os.path.isdir(case_dir) or not openfoam_case:
                subprocess.call(["cp", "-rf", self.dir_basecase, case_dir])

            dir_case_surface = case_dir + "constant/triSurface"
            if not os.path.isdir(dir_case_surface):
                os.mkdir(dir_case_surface)

            # Copy .stl file to the triSurface folder of the case
            if not os.path.isfile(stl_file):
                raise ValueError(
                    "Geometry stl file does not yet exist: run domain generation first"
                )
            else:
                subprocess.call(["cp", stl_file, dir_case_surface])

            # Get location in mesh from stl directory
            lim_file = open(dir_stl + "locationInMesh.dat")
            location_in_mesh = map(float, lim_file.readline().split())
            lim_file.close()

            dir_system = case_dir + "system"
            # Update the blockMeshDict, snappyHexMeshDict and decomposeParDict of the case according to given parameters

            preP.update_blockMeshDict(dir_system, self.domain)

            preP.update_snappyHexMeshDict(
                dir_system,
                self.name_stl_file,
                self.domain["cell_size"],
                location_in_mesh,
                refinement=snappy_refinement,
                castellated_mesh=True,
                snap=snapping if not snappy_refinement else False,
            )
            preP.update_decomposeParDict(dir_system, self.cluster_kwargs['cores_per_sim'])
            preP.update_extrudeMeshDict(dir_system, self.domain["cell_size"])

            writeS.create_pre_processing_script(
                case_dir,
                self.name_run.format(i),
                refinement=snappy_refinement,
                cluster=self.cluster,
                **self.cluster_kwargs
            )

            # Start meshing (when not on cluster)
            if not self.cluster:
                ### ToDo: make it more general by putting file path to a bin folder that is in your $PATH
                os.chdir(case_dir)
                subprocess.run(["chmod","+x","preprocessing{}.sh".format("_0" if snappy_refinement else "")])
                subprocess.run(["./preprocessing{}.sh".format("_0" if snappy_refinement else "")]) 
                #subprocess.run(["chmod","+x","{}preprocessing{}.sh".format(case_dir,"_0" if snappy_refinement else "")])
                #subprocess.run(["sh", "{}preprocessing{}.sh".format(case_dir,"_0" if snappy_refinement else "")]) 

                if snappy_refinement:
                    preP.update_snappyHexMeshDict(
                        dir_system,
                        self.name_stl_file,
                        self.domain["cell_size"],
                        location_in_mesh,
                        refinement=snappy_refinement,
                        castellated_mesh=False,
                        snap=snapping,
                    )
                    subprocess.call(["chmod", "+x", "/preprocessing_1.sh"])
                    subprocess.call(["./preprocessing_1.sh"])
                    #subprocess.run(["chmod", "+x", "{}preprocessing_1.sh".format(case_dir)])
                    #subprocess.run(["sh", "{}/preprocessing_1.sh".format(case_dir)])

                os.chdir(self.python_wd)

        # Start meshing (on cluster)
        if self.cluster:
            waiting_cases = [self.dir_case_sim.format(i) for i in range(self.n_realizations)]
            # waiting_cases=[os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i)) for i in range(n_simulations)]
            finished_cases = []
            active_cases = []

            while len(finished_cases) < self.n_realizations:

                # Start new cases if there are still waiting cases and there are free slots
                while waiting_cases and len(active_cases) < self.cluster_kwargs['n_parallel_sims']:
                    new_case_dir = waiting_cases.pop(0)
                    active_cases.append(new_case_dir)
                    # Put new case in queue (note the '&' which prevents the command from blocking the program)
                    os.chdir(new_case_dir)
                    subprocess.call(["sbatch", "{}/preprocessing{}.sh".format(
                                new_case_dir, "_0" if snappy_refinement else ""),"&"])
                    os.chdir(self.dir_sim)
                    print("Cases running/in queue: {0}".format(active_cases))

                # Give the cluster some time to put the scripts into the queue
                time.sleep(5)

                # Initialize list which will hold the indices of cases which are finished
                end_cases = []
                # Get list of processes that are in the queue or running, together with the username
                queue = os.popen("squeue -h --Format=name:32,username:32,JobID").read()
                queue = queue.split("\n")[:-1]
                # Create dictionary with keys being the names of processes in the queue and values the usernames that started the process
                queue = dict([
                        [i[:32].strip(), [i[32:64].strip(), i[64:].strip()]]
                        for i in queue])
                for i in range(len(active_cases)):
                    # Get references to case directory and case name
                    case_dir = active_cases[i]
                    case_name = case_dir.split(os.sep)[-1]
                    # Check if case is still in the queue (currently username is not taken into consideration)
                    if not "{0}_pre".format(case_name) in queue:
                        if snappy_refinement and not os.path.isfile("{}/snappyHexMesh_1.log".format(case_dir)):
                            # Start second part of meshing if refinement is on and the second snappyHexMesh log has not been created yet
                            os.chdir(case_dir)
                            # Get the stl directory corresponding to this case
                            case_num = case_dir.split("_")[-1]
                            # case_stl_dir = os.path.realpath("{0}{1}{2}_{3}".format(stl_dir, os.sep, run_name, case_num))
                            case_stl_dir = self.dir_stl_files.format(case_num)
                            # Get location in mesh from stl directory
                            lim_file = open("{}/locationInMesh.dat".format(case_stl_dir))
                            location_in_mesh = map(float, lim_file.readline().split())
                            lim_file.close()

                            preP.update_snappyHexMeshDict(
                                case_dir + "system",
                                self.name_stl_file,
                                self.domain["cell_size"],
                                location_in_mesh,
                                refinement=snappy_refinement,
                                castellated_mesh=False,
                                snap=snapping,
                            )  # , snap=True)
                            subprocess.call(["sbatch","{}/preprocessing_1.sh".format(case_dir), "&"])
                            os.chdir(self.dir_sim)
                        else:
                            # Insert the index of finished case at the beginning of the endCases list, so it will be ordered from high to low index
                            end_cases.insert(0, i)

                for i in end_cases:
                    case = active_cases.pop(i)
                    finished_cases.append(case)
                    print("Finished cases: {0}".format(finished_cases))

        print( "\n# Preprocssing for {:d} realizations performed #".format(self.n_realizations))

    def simulate(self, overwrite=False, **cluster_kwargs):

        print("\n### STARTING SIMULATIONS ###\n\n")
        self.cluster_kwargs.update(cluster_kwargs)

        for i in range(self.n_realizations):
            case_dir = self.dir_case_sim.format(i)

            if not os.path.isdir(case_dir):
                print("WARNING: case directory '{}' does not exist! \n \
                    Check preprocessing. \n \
                    Continue by skipping this case.".format(case_dir))
                continue

            if writeS.check_file("{0}/simpleFoam.log".format(case_dir),i=i,overwrite=overwrite,process="Simulation",log=True):
                continue

            writeS.create_simulation_script(
                case_dir, 
                self.name_run.format(i), 
                cluster=self.cluster,
                **cluster_kwargs
            )

            if not self.cluster:
                os.chdir(case_dir)
                subprocess.call(["chmod", "+x", "runSimulations.sh"])
                subprocess.call(["./runSimulations.sh"])
                #subprocess.run(["chmod", "+x", "{}runSimulations.sh".format(case_dir)])
                #subprocess.run(["sh", "{}runSimulations.sh".format(case_dir)])
                writeS.check_file("{}/simpleFoam.log".format(case_dir),i=i,overwrite=True,process="Simulation",log=True)
                os.chdir(self.python_wd)

        print("\n# Simulation of {:d} realizations performed #".format(self.n_realizations))

    def post_process(self,overwrite=False,post_processing_margin=False,file_name_out='out.dat'):
        print("\n### STARTING POST-PROCESSING ###\n\n")
    
        for i in range(self.n_realizations):
            case_dir = self.dir_case_sim.format(i)
            vtk_dir = "{}VTK".format(case_dir) 
            file_output = "{}/{}".format(case_dir,file_name_out)

            if writeS.check_file(file_output,i=i,overwrite=overwrite,process="Postprocessing",log=False):
                continue

            if not os.path.isdir(case_dir) or not os.path.isdir(vtk_dir):
                print("WARNING: sim results do not exist for case {}! \n Check simulation. \
                      \n Continue by skipping this case.".format(i))
                continue

            # Get filename of .vtk file of final iteration, check vtk-files
            vtk_files = []
            for item in os.listdir(vtk_dir):
                if item.lower().endswith(".vtk"): # and os.path.isfile(item):
                    vtk_files.append(item)
            if not vtk_files:
                print("WARNING: no .vtk files in VTK directory of case {}, \
                      skipping this case".format(i))
            vtk_files.sort()
            vtk_file = vtk_files[-1]
            if vtk_file.split("_")[-1] == "0":
                print("WARNING: no .vtk files of timestep > 0 in 'VTK' directory \
                      of case '{}', skipping this case".format(i))
                continue
            
            # Create script for running post-processing on cluster
            if not post_processing_margin:
                post_processing_margin= 0.1*(self.domain["xmax"] - self.domain["xmin"]) *self.unit_factor
            #print(post_processing_margin)

            file_vtk="{}/{}".format(vtk_dir,vtk_file)

            if not self.cluster:
                postP.post_process(file_vtk,file_output,margin=post_processing_margin)

            #writeS.create_post_processing_script(case_dir,filename,"{}/postProcessing.py".format(self.python_wd),  "VTK/{}".format(vtk_file), margin=post_processing_margin)

        #os.chdir(self.dir_sim)
        if self.cluster:
            waiting_cases = [self.dir_case_sim.format(i) for i in range(self.n_realizations)]
            active_cases = []
            finished_cases = []
    
            while len(finished_cases) < self.n_realizations:
                # Start new cases if there are still waiting cases and there are free slots
                while waiting_cases and len(active_cases) < self.cluster_kwargs['n_parallel_sims']:
                    new_case_dir = waiting_cases.pop(0)
                    if not os.path.isfile("{}/postprocessing.sh".format(new_case_dir)):
                        print("WARNING: no 'postprocessing.sh' bash script in case '{0}', skipping case".format(
                                new_case_dir.split(os.sep)[-1]))
                        finished_cases.append(new_case_dir)
                        continue
                    active_cases.append(new_case_dir)
                    # Put new case in queue (note the '&' which prevents the command from blocking the program)
                    os.chdir(new_case_dir)
                    subprocess.call(["sbatch","{0}/postprocessing.sh".format(new_case_dir), "&"])
                    
                    os.chdir(self.dir_sim)
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
#            print("All cases finished")

        print("\n# Postprocessing of {:d} realizations performed #".format(self.n_realizations))

    def write_results(self,file_name_results="results.dat",file_name_out="out.dat"):

            # set results file 
        if "file_results" in self.dirs:
            file_results = self.dirs["file_results"]
        else:
            file_results = "{}{}".format(self.dir_sim,file_name_results)

        if not os.path.isfile(file_results):
            results = open(file_results, "w")
            results.write("Simulation\tPorosity_[-]\tPermeability_[m^2]\n")
        else:
            results = open(file_results, "a+")

        for i in range(self.n_realizations):
            case_dir = self.dir_case_sim.format(i)
            file_output = "{}/{}".format(case_dir,file_name_out)
       
            if not os.path.isdir(case_dir) or not os.path.isfile(file_output):
                print("WARNING: sim result do not exist for case {}! \n Check simulations and postprocessing. \n Continue by skipping this case.".format(i))
                continue

            out = open(file_output, "r")
            por, k = map(float, out.readline().strip().split(","))
            out.close()
    
            results.write("{0}\t{1}\t{2}\n".format(i, por, k))
      
        results.close()
        print('\n# Collection of output for {:d} realizations performed #'.format(self.n_realizations))

###############################################################################
###############################################################################
###############################################################################

grain_model = dict(
    model_type="trunc_lognormal",
    # model_data = dict(rmin=0.00063,rmax=1.0,rmean=0.236,rstd=0.26),
    model_data=dict(rmin=0.005, rmax=0.8, rmean=0.35, rstd=0.25),
    mindist=0.025,
    seed=True,
    boundary_cond="periodic",
)

# grain_model2=dict(model_type = "data_distribution",
#                 model_data = dict(data_points=[0.016, 0.032, 0.05, 0.063, 0.125, 0.25, 0.5, 1.0, 2.0],
#                                   c_freq=[0, 1, 3.5, 3.7, 5.0, 14, 68, 94, 100]),
#                 mindist = 0.025,
#                 seed = True)

domain = dict(
    xmin=0,
    xmax=10,
    ymin=0,
    ymax=5,
    por_min=0.3,
    por_max=0.4,
    cell_size=grain_model["mindist"]/np.sqrt(8),
    unit='mm',
)

# from pathlib import Path
path = os.getenv("HOME") + "/Projects/GrainSizeAnalysis/Workflow/"
name="Ensemble_02"
# path = os.getcwd()

dirs = dict(
#    dir_root = path,
    name_run = name,  # Name of this batch of simulations
    basecase_symmetry = path + "baseCase_symmetry/",  # Directory to copy the base OpenFOAM case from
    basecase_cyclic= path + "baseCase_cyclic/",
#    file_settings = "{}{}/Settings_{}.txt".format(path, name, name), 
#    file_results = "{}{}/Results_{}.txt".format(path, name, name), 
)

#cluster_kwargs = dict(
#    cores_per_sim=1,
#    n_parallel_sims=12,  # Number of simulations to run in parallel at one time (value higher than 1 only supported for cluster)
#    tasks_per_node=1,  # Amount tasks to be invoked per computing node
#    threads_per_core=2,  # Restrict node selection to nodes with at least the specified number of threads per core.
#    partition="allq",  # Which queue to run simulations on
#    modules = ["opt/all",           
#               "gcc/6.4.0",
#               "openmpi/gcc-6.4.0/3.1.2",
#               "openFoam/6"],
#    scripts=[ # List of scripts to source when running bash script on cluster
#        "/trinity/opt/apps/software/openFoam/version6/OpenFOAM-6/etc/bashrc"
#    ],  
#)

#cluster_new=dict(cores_per_sim=2)
#cluster_kwargs.update(cluster_new)

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### Initialize Ensemble
E1 = Ensemble(
    grain_model, 
    domain, 
    dirs, 
    n_realizations=5,
    cluster=False,
)  # , pre_process = True, simulations = True, post_process = True)
#from vtkTools import VTKObject
#
#file_vtk='/home/zech0001/Projects/GrainSizeAnalysis/Workflow/field_data/cases/field_data_0/VTK/field_data_0_88.vtk'
#vtk = VTKObject(file_vtk, calc_volumes=True)
#test=vtk.calc_mean("p")
#
###############################################################################
### Initializing structure and settings (done internally)

# E1.setup_dir_structure()
# E1.set_seed()
# E1.write_settings()

###############################################################################
### Run Workflow on Ensemble - do explicitely

E1.generate_geometries()
E1.pre_process() #, **cluster_kwargs)
E1.simulate()
E1.post_process()
E1.write_results()
