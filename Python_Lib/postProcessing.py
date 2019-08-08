"""postProcessing.py: some tools to post process an OpenFOAM simulation."""

import sys
import pickle
import numpy as np
#from vtk import vtkUnstructuredGridReader
#from vtkTools import _unst_grid_read
#from scipy.spatial import ConvexHull

### Local dependencies
from vtkTools import VTKObject

def calc_permeability(por, u_mean, dp, dx, kin_visc=1e-6, density=1000):
    """Calculate the permeability from Darcy's law:
        k_x = - kin_visc * por * U_mean * density * dx / dp
        
        PARAMETERS
        ----------
        por : float, int
            Porosity value to be used in calculation.
        u_mean : float, int
            Mean velocity along the direction for which the permeability is being calculated.
        kin_visc : float, int
            Kinematic viscosity of fluid.
        density : float, int
            Density of fluid.
        dp : float, int
            Pressure gradient over the length over which permeability is being calculated.
        dx : float, int
            Length over which permeability is being calculated.
        
        RETURNS
        -------
        k : float
            Permeability calculated using Darcys law."""

    k = (-1.0 * float(por) * float(kin_visc)* float(u_mean) * float(density) * float(dx) / float(dp) )

    return k

def calc_values(vtk,  margin=0, epsilon=0.00001,**kwargs):
    """Do post-processing on VTKObject, calculating the porosity and permeability of the domain
        given a certain margin which is excluded from the calculation.
        
        PARAMETERS
        ----------
        vtk : VTKObject
            VTKObject to do calculations on.
        kin_visc : float, int
            Kinematic viscosity of fluid.
        density : float, int
            Density of fluid.
        margin : float, int
            Margin to be excluded from calculations at the inlet and outlet of the domain.
        epsilon : float
            Range around the minimum and maximum x values of the domain (excluding the margins) around which cells will be used to calculate the pressure gradient,
            e.g. for epsilon = 0.0001, the pressure on the left side of the domain will be calculated as the mean of the cells between x = (xmin + margin) - 0.0001, and x = (xmin + margin) + 0.0001
        
        RETURNS
        -------
        por : float
            Porosity calculated for the VTKObject with given margin.
        k : float
            Permeability calculated for the VTKObject with given margin.
    """

    if "U_x" not in vtk.cell_data:
        # Write vector components of velocity
        vtk.write_vector_components("U")

    # Define box for which permeability will be determined
    box = [vtk.xmin + margin, vtk.xmax - margin, vtk.ymin, vtk.ymax, vtk.zmin, vtk.zmax]
    # Calculate mean pressures at both ends of the region
    p1 = vtk.calc_mean( "p",
        region=[box[0] - epsilon, box[0] + epsilon, box[2], box[3], box[4], box[5]],
        volume_weighted=True)
    p2 = vtk.calc_mean( "p",
        region=[box[1] - epsilon, box[1] + epsilon, box[2], box[3], box[4], box[5]],
        volume_weighted=True)
    # Calculate permeability of the region
    por = vtk.calc_porosity(region=box)
    # Calculate mean velocity in x-direction in the region
    Ux_mean = vtk.calc_mean("U_x", region=box, volume_weighted=True)
    # Calculate permeability from porosity, weighted velocity, kinematic viscosity, density, pressure gradient and
    # region length
    k = calc_permeability(por, Ux_mean, p2 - p1, box[1] - box[0], **kwargs)

    return por, k

###############################################################################
### Additional Funcions (not used in postprocessing yet)
###############################################################################

def volume_weighted_mean(values, volumes):
    """Calculates the mean of the values array, each value weighted by the corresponding index of the volumes array.

        PARAMETERS
        ----------
        values : array_like
            Array of values of which the mean value will be calculated.
        volumes : array_like
            Array of volumes corresponding to the values array.
        
        RETURNS
        -------
        weighted_mean : float
            Volume weighted mean of values array."""

    if len(values) > len(volumes):
        # Not all values can be weighted if there are more values than volumes
        print("ERROR: array of values if longer than array of volumes")
        return
    if len(volumes) > len(volumes):
        # Only use the corresponding indices of the volumes array if too many volumes are provided
        print(
            "WARNING: array of volumes is longer than array of values, not all volumes will be used"
        )

    weighted_sum = 0
    for i in range(len(values)):
        weighted_sum += values[i] * volumes[i]

    weighted_mean = weighted_sum / np.sum(volumes)

    return weighted_mean

def analyse_rev(vtk_obj, step, margin,**kwargs):
    """Calculate porosities and permeabilities of a VTKObject instance over a range of regions within the domain.
        The regions that are analysed are determined by a step and a margin. Starting from a region with volume of 0 at the center of the domain,
        every calculation the region is extended in the x- and y-direction by the step size. When the region comes within a margin 
        (given as percentage of the entire domain size) the calculation is stopped.
        
        PARAMETERS
        ----------
        vtk_obj : VTKObject
            VTKObject to do REV calculation on.
        step : float, int
            How much is added to the calculation domain each calculation.
        margin : float, int
            Margin to be excluded from calculations on all sides of the domain

        **kwargs:
            kin_visc : float, int
                Kinematic viscosity of fluid.
            density : float, int
                Density of fluid.
    """

    # Check if the VTKObject instance contains cell volumes and the x component
    if not hasattr(vtk_obj, "cell_volumes"):
        vtk_obj.calc_cell_volumes()
    if "U_x" not in vtk_obj.cell_data:
        if "U" not in vtk_obj.cell_data:
            print("ERROR: VTKObject instance does not contain velocity cell data")
            return
        else:
            vtk_obj.write_vector_components("U", data_type="cell")

    # Calculate some variables to be used
    xmin, xmax, ymin, ymax, zmin, zmax = (
        vtk_obj.xmin,
        vtk_obj.xmax,
        vtk_obj.ymin,
        vtk_obj.ymax,
        vtk_obj.zmin,
        vtk_obj.zmax,
    )
    center_x = xmin + 0.5 * xmax
    center_y = ymin + 0.5 * ymax
    dx = xmax - xmin
    dy = ymax - ymin

    # Open the file to write porosities and permeabilities
    file = open("REV_analysis.dat", "w")
    file.write("dx\tdy\tdz\tporosity\tdP\tpermeability\n")

    box = [center_x, center_x, center_y, center_y, zmin, zmax]
    while True:
        # Expand box by step in x and y direction
        box = [
            box[0] - step / 2,
            box[1] + step / 2,
            box[2] - step / 2,
            box[3] + step / 2,
            zmin,
            zmax,
        ]

        if (
            box[0] > xmin + margin * dx
            and box[1] < xmax - margin * dx
            and box[2] > ymin + margin * dy
            and box[3] < ymax - margin * dy
        ):
            print(
                "Calculating porosity and permeability for box x: {0} -> {1}; y: {2} -> {3}; z: {4} -> {5}".format(
                    box[0], box[1], box[2], box[3], box[4], box[5]
                )
            )

            # Calculate porosity and mean velocity along x-axis in the current region
            por = vtk_obj.calc_porosity(region=box)
            ux_mean = vtk_obj.calc_mean("U_x", region=box, volume_weighted=True)

            # Calculate the pressure gradient over the region
            # First calculate mean pressure over the xmin plane of the region
            p1 = (
                vtk_obj.calc_mean(
                    "p",
                    region=[
                        box[0] - 0.001 * dx,
                        box[0] + 0.001 * dx,
                        box[2],
                        box[3],
                        box[4],
                        box[5],
                    ],
                    volume_weighted=True,
                )
                * density
            )
            # Secondly calculate mean pressure over the xmax plane of the region
            p2 = (
                vtk_obj.calc_mean(
                    "p",
                    region=[
                        box[1] - 0.001 * dx,
                        box[1] + 0.001 * dx,
                        box[2],
                        box[3],
                        box[4],
                        box[5],
                    ],
                    volume_weighted=True,
                )
                * density
            )

            k = calc_permeability(por, ux_mean, p2 - p1, box[1] - box[0], **kwargs)

            file.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                    box[1] - box[0], box[3] - box[2], box[5] - box[4], por, p2 - p1, k
                )
            )
        else:
            break

    file.close()


def plot_rev_data(path):
    """Plots data (structured as written by the analyseREV function: dx, dy, dz, porosity, dP, permeability), to show change in porosity
        and permeability with different volumes.
        
        PARAMETERS
        ----------
        path : str
            Path of the data to analyse."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Failed to import matplotlib.pyplot, exiting.")
        return
    try:
        from scipy.interpolate import make_interp_spline #, BSpline
    except ImportError:
        print(
            "Failed to import make_inter_spline and/or BSpline from scipy.interpolate, exiting."
        )
        return

    file = open(path, "r")
    header = file.readline().split()

    dx_idx = header.index("dx")
    dy_idx = header.index("dy")
    dz_idx = header.index("dz")
    por_idx = header.index("porosity")
    k_idx = header.index("permeability")

    u = []
    por = []
    k = []

    for line in file.readlines():
        line = line.split()
        if line:
            u.append(float(line[dx_idx]) * float(line[dy_idx]) * float(line[dz_idx]))
            por.append(float(line[por_idx]))
            k.append(float(line[k_idx]))

    u_smooth = np.linspace(u[0], u[-1], 10000)
    por_spl = make_interp_spline(u, por, k=3)
    por_smooth = por_spl(u_smooth)
    k_spl = make_interp_spline(u, k, k=3)
    k_smooth = k_spl(u_smooth)

    plt.plot(u, por, ".", label="Porosity", color="black")
    plt.plot(u_smooth, por_smooth, "-", color="black")
    plt.xlabel("Volume [m$^3$]")
    plt.ylabel("Porosity [-]")
    plt.title("Sample volume vs. Porosity")
    plt.show()

    plt.plot(u, k, ".", label="Permeability", color="black")
    plt.plot(u_smooth, k_smooth, "-", color="black")
    plt.xlabel("Volume [m$^3$]")
    plt.ylabel("Permeability [m$^2$]")
    plt.yscale("log")
    plt.title("Sample volume vs. Permeability")
    plt.show()

def load_vtk_object(path):
    """Load VTKObject instance from a pickled file. Returns the object.

        PARAMETERS
        ----------
        path : str
            Path to the file to load.
        
        RETURNS
        -------
        vtk_obj : VTKObject
            The VTKObject instances loaded from the pickled file."""

    file = open(path, "br")
    vtk_obj = pickle.load(file)
    file.close()

    return vtk_obj

def ensemble_statistics(data, bins, file=None):
    """Get statistics from ensemble results sorted into porosity ranges.

        PARAMETERS
        ----------
        data : array_like
            Array of ensemble data organized as couples of porosities with corresponding permeability,
            i.e.: [[porosity_0, permeability_0], [porosity_1, permeability_1], ..., [porosity_last, permeability_last]].
        bins : array_like
            Array defining the porosity ranges to calculate statistics for. For example,
            [0.3, 0.35, 0.4] will calculate statistics for porosity ranges (0.3, 0.35) and (0.35, 0.4).
        file : str
            If an output file is desired, this should be the file's path.
        
        RETURNS
        -------
        mean : list
            List of mean permeabilities per porosity range.
        std : list
            List of standard deviations of permeability per porosity range."""

    data = np.array(data)
    mean = []
    std = []

    for i in range(len(bins) - 1):
        data_filtered = np.compress(
            (bins[i] < data[:, 0]) * (bins[i + 1] > data[:, 0]), data[:, 1]
        )
        mean.append(np.mean(data_filtered))
        std.append(np.std(data_filtered))

    if file:
        out_file = open(file, "w")
        out_file.write("por_bin\tk_mean\tk_std\n")
        for i in range(len(mean)):
            out_file.write(
                "{0}-{1}\t{2}\t{3}\n".format(bins[i], bins[i + 1], mean[i], std[i])
            )
        out_file.close()

    return mean, std

###############################################################################
### Calling main for postprocessing on the cluster
###############################################################################

#def post_process(case_dir,file_vtk,**kwargs):
def post_process(file_vtk,file_output,**kwargs):
    
    vtk = VTKObject(file_vtk, calc_volumes=True)
    por, k = calc_values(vtk,**kwargs)

    print("Writing output to file: \n {}".format(file_output))  
    outfile = open(file_output, "w")
    outfile.write("{0},{1}".format(por, k))
    outfile.close()

if __name__ == "__main__":
    case_dir = sys.argv[1]
    filename = sys.argv[2]

    if len(sys.argv) > 3:
        margin = sys.argv[3]
    else:
        margin = 0
    if len(sys.argv) > 4:
        kin_visc = sys.argv[4]  
    else:
        kin_visc = 1e-6
    if len(sys.argv) > 5:
        density = sys.argv[5]
    else:
        density = 1000

    post_process(case_dir,filename, margin ,kin_visc, density)
        