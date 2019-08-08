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
        bc="periodic",
    ):

        self.grain_model = grain_model
        self.domain = domain
        self.dirs = dirs
        self.n_realizations = n_realizations
        self.bc = bc

        self.setup_dir_structure()
        self.set_seed()
        self.write_settings()

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

        # Check if baseDir path already exists, if not, create it
        self.dir_sim = self.dir_root + self.name_run
        if not os.path.isdir(self.dir_sim):
            os.mkdir(self.dir_sim)
        # os.chdir(self.dirs['dir_sim'])

        self.dir_stl = self.dir_sim + "/stl"
        if not os.path.isdir(self.dir_stl):
            os.mkdir(self.dir_stl)

        self.dir_stl_files = self.dir_stl + "/" + self.name_run + "_{}"
        self.name_stl_file = "stl"

        self.dir_cases = self.dir_sim + "/cases"
        if not os.path.isdir(self.dir_cases):
            os.mkdir(self.dir_cases)
        self.dir_case_sim = self.dir_cases + "/" + self.name_run + "_{}"

        if self.bc in ["periodic", "cyclic"]:
            dir_base = self.dirs["basecase_cyclic"]
        else:
            dir_base = self.dirs["basecase_symmetry"]

        self.dir_basecase = self.dir_sim + "/baseCase"
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

        self.python_wd = os.getcwd()

    def set_seed(self, seed=False):

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

    def generate_geometries(self, stl_overwrite=False):

        """

        Optional Arguments
        ------------------
        stl_overwrite       - Whether or not to overwrite stl files if they already exist

        """

        print("\n### STARTING DOMAIN GENERATION ###\n\n")

        for i in range(self.n_realizations):

            if not os.path.isdir(self.dir_stl_files.format(i)):
                os.mkdir(self.dir_stl_files.format(i))

            file_stl = self.dir_stl_files.format(i) + "/" + self.name_stl_file + ".stl"

            if (not os.path.isfile(file_stl)) or (
                os.path.isfile(file_stl) and stl_overwrite
            ):
                if os.path.isfile(file_stl) and stl_overwrite:
                    print("Overwriting stl file '{}'".format(file_stl))

                if (
                    self.seed
                ):  # If seed is in use, update it for this specific simulation
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
                            "WARNING: provided seed was unable to generate correct geometry, random seed will be used for case {0}".format(
                                i
                            )
                        )
                        self.grain_model["seed"] = np.random.randint(0, 10 ** 9)
                    por = randomCircles.create_model(
                        self.grain_model,
                        self.domain,
                        stl_filename=self.name_stl_file,
                        path=self.dir_stl_files.format(i),
                    )
            else:
                print("stl file '{}' already exists, not overwritten".format(file_stl))

        print(
            "\n### Domains for {:d} realizations successfully set-up".format(
                self.n_realizations
            )
        )

    def pre_process(
        self,
        snapping=True,
        snappy_refinement=False,
        cluster=False,
        overwrite=False,
        cores_per_sim=1,
        n_parallel_sims=12,
        **cluster_kwargs
    ):  

        """
        Optional Arguments
        ------------------
        snapping            - Whether or not to perform mesh snapping to grain surfaces during pre-processing                          
        snappy_refinement   - Whether or not to perform mesh refinement around grains during pre-processing
        cluster             - Whether or not to run on cluster
        cores_per_sim       - Number of cores to use for each simulation
        n_parallel_sims     - 
        
        cluster_args        - cluster arguments
            tasks_per_node      - Amount tasks to be invoked per computing node
            threads_per_core    - Restrict node selection to nodes with at least the specified number of threads per core.
            partition           - Which queue to run simulations on
            modules             - List of modules to load when running bash script on cluster
            scripts             - List of scripts to source when running bash script on cluster
        """

        print("\n### STARTING PRE-PROCESSING ###\n\n")

        if self.bc in ["periodic", "cyclic"] and cores_per_sim > 1:
            print(
                "### WARNING:  using multiple cores for a case with cyclic BC is error prone"
            )

        for i in range(self.n_realizations):

            # Get reference to directory of this case, this case's stl directory and to this case's .stl file
            case_dir = self.dir_case_sim.format(i)
            dir_stl = self.dir_stl_files.format(i) + "/"
            stl_file = dir_stl + self.name_stl_file + ".stl"

            if preP.check_log("{0}/snappyHexMesh_0.log".format(case_dir)):
                if not overwrite:
                    print(
                        " Preprocessing for case {} already performed  \
                                \n continue with next case.".format(i))
                    continue
                else:
                    print(
                        " Preprocessing for case {} already performed \n \
                            preprocessing repeated, file overwritten.".format(i))

            # Create case from baseCase if it does not exist yet, or if existing directory is not a valid OpenFOAM directory
            openfoam_dirs = [
                "{}/{}".format(case_dir, folder)
                for folder in ["0", "constant", "system"]
            ]
            openfoam_case = np.array(
                [os.path.isdir(folder) for folder in openfoam_dirs]
            ).all()
            if not os.path.isdir(case_dir) or not openfoam_case:
                subprocess.call(["cp", "-rf", self.dir_basecase, case_dir])

            dir_case_surface = case_dir + "/constant/triSurface"
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

            dir_system = case_dir + "/system"
            # Update the blockMeshDict, snappyHexMeshDict and decomposeParDict of the case according to given parameters

            preP.update_blockMeshDict(
                dir_system, self.domain
            )

            preP.update_snappyHexMeshDict(
                dir_system,
                self.name_stl_file,
                self.domain["cell_size"],
                location_in_mesh,
                refinement=snappy_refinement,
                castellated_mesh=True,
                snap=snapping if not snappy_refinement else False,
            )
            preP.update_decomposeParDict(dir_system, cores_per_sim)
            preP.update_extrudeMeshDict(dir_system, domain["cell_size"])

            writeS.create_pre_processing_script(
                case_dir,
                self.name_run.format(i),
                refinement=snappy_refinement,
                cluster=cluster,
                **cluster_kwargs
            )

            # Start meshing (when not on cluster)
            if not cluster:
                ### ToDo: make it more general by putting file path to a bin folder that is in your $PATH
                os.chdir(case_dir)
                subprocess.run(
                    [
                        "chmod",
                        "+x",
                        "preprocessing{}.sh".format("_0" if snappy_refinement else ""),
                    ]
                )
                # subprocess.Popen(['sh',"./preprocessing{}.sh".format("_0" if snappy_refinement else "")])#,shell=True)
                subprocess.call(
                    ["./preprocessing{}.sh".format("_0" if snappy_refinement else "")]
                )  # ,shell=True)
                # print("{}/preprocessing{}.sh".format(case_dir,"_0" if snappy_refinement else ""))
                # print(os.path.isfile("{}/preprocessing{}.sh".format(case_dir,"_0" if snappy_refinement else "")))

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
                    # subprocess.Popen(['sh',"./preprocessing_1.sh"])
                    subprocess.call(["./preprocessing_1.sh"])

                os.chdir(self.python_wd)

        # Start meshing (on cluster)
        if cluster:

            waiting_cases = [
                self.dir_case_sim.format(i) for i in range(self.n_realizations)
            ]
            #            waiting_cases=[os.path.realpath("cases{0}{1}_{2}".format(os.sep, run_name, i)) for i in range(n_simulations)]
            finished_cases = []
            active_cases = []

            while len(finished_cases) < self.n_realizations:

                # Start new cases if there are still waiting cases and there are free slots
                while waiting_cases and len(active_cases) < n_parallel_sims:
                    new_case_dir = waiting_cases.pop(0)
                    active_cases.append(new_case_dir)
                    # Put new case in queue (note the '&' which prevents the command from blocking the program)
                    os.chdir(new_case_dir)
                    subprocess.call(
                        [
                            "sbatch",
                            "{0}{1}preprocessing{2}.sh".format(
                                new_case_dir, os.sep, "_0" if snappy_refinement else ""
                            ),
                            "&",
                        ]
                    )
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
                queue = dict(
                    [
                        [i[:32].strip(), [i[32:64].strip(), i[64:].strip()]]
                        for i in queue
                    ]
                )
                for i in range(len(active_cases)):
                    # Get references to case directory and case name
                    case_dir = active_cases[i]
                    case_name = case_dir.split(os.sep)[-1]
                    # Check if case is still in the queue (currently username is not taken into consideration)
                    if not "{0}_pre".format(case_name) in queue:
                        if snappy_refinement and not os.path.isfile(
                            "{0}{1}snappyHexMesh_1.log".format(case_dir, os.sep)
                        ):
                            # Start second part of meshing if refinement is on and the second snappyHexMesh log has not been created yet
                            os.chdir(case_dir)

                            # Get the stl directory corresponding to this case
                            case_num = case_dir.split("_")[-1]
                            # case_stl_dir = os.path.realpath("{0}{1}{2}_{3}".format(stl_dir, os.sep, run_name, case_num))
                            case_stl_dir = self.dir_stl_files.format(case_num)

                            # Get location in mesh from stl directory
                            lim_file = open(
                                "{0}/locationInMesh.dat".format(case_stl_dir)
                            )
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
                            subprocess.call(
                                [
                                    "sbatch",
                                    "{0}{1}preprocessing_1.sh".format(case_dir, os.sep),
                                    "&",
                                ]
                            )
                            os.chdir(self.dir_sim)
                        else:
                            # Insert the index of finished case at the beginning of the endCases list, so it will be ordered from high to low index
                            end_cases.insert(0, i)

                for i in end_cases:
                    case = active_cases.pop(i)
                    finished_cases.append(case)
                    print("Finished cases: {0}".format(finished_cases))

        print(
            "Preprocssing for {:d} realizations successfully".format(
                self.n_realizations
            )
        )

    def simulate(self, overwrite=False, cluster=False, **cluster_kwargs):

        print("\n### STARTING SIMULATIONS ###\n\n")

        for i in range(self.n_realizations):
            case_dir = self.dir_case_sim.format(i)
            if not os.path.isdir(case_dir):
                print(
                    "WARNING: case directory '{}' does not exist! \n Check preprocessing. \n Continue by skipping this case.".format(
                        case_dir
                    )
                )
                continue
            if preP.check_log("{0}/simpleFoam.log".format(case_dir)):
                if not overwrite:
                    print(
                        " Simulation for case {} already performed  \
                                \n continue with next case.".format(i))
                    continue
                else:
                    print(
                        " Simulation for case {} already performed \n \
                            Simulation repeated, output overwritten.".format(i))

            writeS.create_simulation_script(
                case_dir, self.name_run.format(i), **cluster_kwargs
            )

            if not cluster:
                os.chdir(case_dir)
                subprocess.call(["chmod", "+x", "runSimulations.sh"])
                subprocess.call(["./runSimulations.sh"])
                if not preP.check_log("{0}/simpleFoam.log".format(case_dir)):
                    print("Case '{0}' failed to run properly.".format(i))
                #                    print("Restarting case '{0}'".format(case_name))
                os.chdir(self.python_wd)

        print(
            "Simulation of {:d} realizations successfully".format(self.n_realizations)
        )

    def post_process(self,cluster=False,post_processing_margin=False):
        print("\n### STARTING POST-PROCESSING ###\n\n")
    
        for i in range(self.n_realizations):
            case_dir = self.dir_case_sim.format(i)
            vtk_dir = "{}/VTK".format(case_dir) 

            if not os.path.isdir(case_dir) or not os.path.isdir(vtk_dir):
                print("WARNING: sim results does not exist! \n Check simulation. \n Continue by skipping this case.")
                continue

            # Get filename of .vtk file of final iteration, check vtk-files
            vtk_files = []
            for item in os.listdir(vtk_dir):
                if item.lower().endswith(".vtk"): # and os.path.isfile(item):
                    vtk_files.append(item)
            if not vtk_files:
                print("WARNING: no .vtk files in VTK directory of case {}, skipping this case".format(i))
            vtk_files.sort()
            vtk_file = vtk_files[-1]
            if vtk_file.split("_")[-1] == "0":
                print("WARNING: no .vtk files of timestep > 0 in 'VTK' directory of case '{}', skipping this case".format(i))
                continue
    
            # Create script for running post-processing on cluster
            if not post_processing_margin:
                post_processing_margin=(self.domain["xmax"] - self.domain["xmin"]) * 0.1


            file_vtk="{}/{}".format(vtk_dir,vtk_file)

            if "file_output" in self.dirs:
                file_output = self.dirs["file_output"]
            else:
                file_output = "{}/out.dat".format(self.dir_sim)

            if not cluster:
                postP.post_process(file_vtk,file_output,margin=post_processing_margin)

            #writeS.create_post_processing_script(case_dir,filename,"{}/postProcessing.py".format(self.python_wd),  "VTK/{}".format(vtk_file), margin=post_processing_margin)













        print(
            "Postprocessing of {:d} realizations successfully".format(
                self.n_realizations
            )
        )


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
)

# from pathlib import Path
path = os.getenv("HOME") + "/Projects/GrainSizeAnalysis/Workflow/"
name="Ensemble_02"
# path = os.getcwd()

dirs = dict(
    dir_root = path,
    name_run = name,  # Name of this batch of simulations
    basecase_symmetry = path + "baseCase_symmetry",  # Directory to copy the base OpenFOAM case from
    basecase_cyclic= path + "baseCase_cyclic",
#    file_settings = path + "Settings_{}.txt".format(name), 
#    file_output = path + "Output.txt",
)

cluster_kwargs = dict(
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
        "/trinity/opt/apps/software/openFoam/version6/OpenFOAM-6/etc/bashrc"
    ],  
)

# def test(**kwargs):
#    for key, value in kwargs.items():
#        print ("{} == {}".format(key,value))
#
# test(**cluster_kwargs)

# Which steps to perform
# generate_domain = True
# pre_process = True
# simulate = True
# post_process = True

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### Initialize Ensemble
E1 = Ensemble(
    grain_model, domain, dirs, n_realizations=2
)  # , pre_process = True, simulations = True, post_process = True)

###############################################################################
### Initializing structure and settings (done internally)

# E1.setup_dir_structure()
# E1.set_seed()
# E1.write_settings()

###############################################################################
### Run Workflow on Ensemble - do explicitely

#E1.generate_geometries()
#E1.pre_process(cluster=False) #, **cluster_kwargs)
#E1.simulate(cluster=False)
E1.post_process(cluster=False)

