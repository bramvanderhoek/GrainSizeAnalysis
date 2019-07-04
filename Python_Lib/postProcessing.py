"""postProcessing.py: some tools to post process an OpenFOAM simulation."""

import os, sys
import pickle
import numpy as np
from vtk import vtkUnstructuredGridReader
from vtkTools import _unst_grid_read
from scipy.spatial import ConvexHull
from preProcessing import create_script_header


class VTKObject:
    """Represents the data inside a VTK file, includes methods to do post-processing with the data."""

    def __init__(self, file_path, calc_volumes=False):
        # Attempt to read VTK file from filepath
        if not os.path.isfile(file_path):
            print("ERROR: {0} is not a file".format(file_path))
        elif not file_path.lower().endswith(".vtk"):
            print("ERROR: {0} does not have the proper file extension (.vtk)".format(file_path))
        else:
            print("Opening input file {0}".format(file_path))
            # Read in data from vtk file
            data = read_vtk(file_path)
            # Get points and point data
            self.points = data["points"]
            self.point_data = data["point_data"]
            self.n_points = len(self.points)

            # Get cells and cell data
            cells = data["cells"]
            cell_data = data["cell_data"]
            keys = [key for key in cells]
            attrs = [attr for attr in cell_data[keys[0]]]
            # Reorder cells and celldata by cellID
            self.n_cells = np.sum([len(cells[key]) for key in keys])
            self.cell_points = np.ndarray(self.n_cells, dtype=np.ndarray)
            self.cell_data = dict([[attr, np.ndarray(self.n_cells, dtype=type(cell_data[keys[0]][attr]))] for attr in attrs])
            for i in range(self.n_cells):
                for key in keys:
                    if len(cell_data[key]["cellID"]) > 0 and cell_data[key]["cellID"][0] == i:
                        self.cell_points[i], cells[key] = cells[key][0], cells[key][1:]
                        for attr in attrs:
                            self.cell_data[attr][i], cell_data[key][attr] = cell_data[key][attr][0], cell_data[key][attr][1:]
                        break

            # Get extents
            x = np.array([point[0] for point in self.points])
            y = np.array([point[1] for point in self.points])
            z = np.array([point[2] for point in self.points])
            self.xmin = np.min(x)
            self.xmax = np.max(x)
            self.ymin = np.min(y)
            self.ymax = np.max(y)
            self.zmin = np.min(z)
            self.zmax = np.max(z)

            if calc_volumes:
                self.calc_cell_volumes()

    def calc_cell_volumes(self):
        """Calculate the volume of each cell (in m^3) and save it into the cellVolumes array.
Uses the ConvexHull class in the scipy.spatial package (which might not be the fastest way to do it).
Note that this therefore also assumes that the volume of the cell is equal to the volume of the convex hull
resulting from the cell's points, which would not be the case if the cell is concave."""

        print("Calculating cell volumes for {0} cells".format(self.n_cells))
        # Initialize the array to hold the volumes
        self.cell_volumes = np.ndarray(self.n_cells, dtype=float)
        for i in range(self.n_cells):
            # Get the pointIds for the current cell
            point_ids = self.cell_points[i]
            # Create a convex hull from the cell's points 
            hull = ConvexHull([self.points[idx] for idx in point_ids])
            # Get the volume of the convex hull and write it to the array
            self.cell_volumes[i] = hull.volume

    def calc_cell_centers(self):
        """Calculates cell centers as the mean of its extrema:
center = (mean(xmax, xmin), mean(ymax, ymin), mean(zmax, zmin))""" 

        self.cell_centers = np.ndarray(self.n_cells, dtype=np.ndarray)
        for i in range(self.n_cells):
            # Get pointIds of current cell
            point_ids = self.cell_points[i]
            # Get list of coordinates of point on separate axes
            x = [self.points[idx][0] for idx in point_ids]
            y = [self.points[idx][1] for idx in point_ids]
            z = [self.points[idx][2] for idx in point_ids]
            # Get minimum and maximum extents of the cell on all three axes
            xmax, xmin = np.max(x), np.min(x)
            ymax, ymin = np.max(y), np.min(y)
            zmax, zmin = np.max(z), np.min(z)
            # Calculate cell center as the point defined by the mean of the minimum and maximum value of each axis
            center = np.array([np.mean([xmax, xmin]), np.mean([ymax, ymin]), np.mean([zmax, zmin])])
            self.cell_centers[i] = center

    def calc_mean(self, data, region=None, volume_weighted=False):
        """Calculate the mean value of data. If region is specific (as list: [xmin, xmax, ymin, ymax, zmin, zmax]),
calculate mean only over cells included in this region. If volumeWeighted is true, weigh every value by the volume of
the cell.

PARAMETERS
----------
data : str
    Name of the data to take the mean value of.
region : array_like
    Region over which to take the mean.
volume_weighted : bool
    Whether or not to weigh the mean by volume of cells.

RETURNS
-------
mean : float
    Mean value calculated from the data."""

        # Check if region argument has proper length, otherwise raise error.
        if region and not len(region) == 6:
            print("ERROR: provided argument 'region' does not have length of 6. 'region' should be a list with structure [xmin, xmax, ymin, ymax, zmin, zmax]")
            return 

        # Calculated volumes if volume weighted is enabled but instance does not have a cellVolumes attribute
        if volume_weighted and not hasattr(self, "cell_volumes"):
            self.calc_cell_volumes()

        total = 0
        total_cells = 0
        total_volume = 0
        for cell in range(self.n_cells):
            if region:
                in_region = self.cell_in_region(cell, region)
                if not in_region:
                    # Skip point if its not inside of the specified region
                    continue
            if volume_weighted:
                total += self.cell_data[data][cell] * self.cell_volumes[cell]
                total_volume += self.cell_volumes[cell]
            else:
                total += self.cell_data[data][cell]
                total_cells += 1
        if volume_weighted and total_volume > 0:
            mean = total / total_volume
        else:
            mean = total / total_cells

        return mean

    def write_vector_components(self, vector_attribute, data_type="cell"):
        """Split vector up into it's x-, y-, and z-components and write those to the cellData or pointData dictionary.

PARAMETERS
---------
vector_attribute : str
    Name of the vector attribute to split up into its components.
data_type : str
    Which data type to search for ('cell' or 'point')."""

        # Create references according to datatype
        if data_type == "cell":
            data_dict = self.cell_data
            n = self.n_cells
        elif data_type == "point":
            data_dict = self.point_data
            n = self.n_points
        else:
            print("ERROR: data_type should either be 'cell' or 'point'")
            return

        # Check if attribute exists, is an array, and is three dimensional
        if vector_attribute not in data_dict:
            print("ERROR: {0} was not found as a {1} data attribute".format(vector_attribute, data_type))
            return
        if not isinstance(data_dict[vector_attribute][0], np.ndarray):
            print("ERROR: {0} in {1} data is not vector data".format(vector_attribute, data_type))
            return
        if not len(data_dict[vector_attribute][0]) == 3:
            print("ERROR: {0} {1} data are not 3-dimensional vectors".format(vector_attribute, data_type))
            return

        print("Writing x-, y- and z-components of {0} for {1} {2}{3}".format(vector_attribute, n, data_type, "s" if n > 1 else ""))

        # Initialize data arrays
        data_dict["{0}_x".format(vector_attribute)] = np.ndarray(n)
        data_dict["{0}_y".format(vector_attribute)] = np.ndarray(n)
        data_dict["{0}_z".format(vector_attribute)] = np.ndarray(n)

        # Split vectors up into components
        for i in range(n):
            data_dict["{0}_x".format(vector_attribute)][i] = data_dict[vector_attribute][i][0]
            data_dict["{0}_y".format(vector_attribute)][i] = data_dict[vector_attribute][i][1]
            data_dict["{0}_z".format(vector_attribute)][i] = data_dict[vector_attribute][i][2]

    def calc_porosity(self, region=None):
        """Calculate porosity by assuming the model is defined by a box determined by the minimum and maximum extent,
and dividing the sum of the cell volumes by the total volume of this box.

PARAMETERS
----------
region : array_like
    Array of the region to calculate the porosity for as [xmin, xmax, ymin, ymax, zmin, zmax].

RETURNS
-------
porosity : float
    Calculated porosity."""

        if not hasattr(self, "cell_volumes"):
            self.calc_cell_volumes()
        if region and not len(region) == 6:
            print("ERROR: provided argument 'region' does not have length of 6. 'region' should be a list with "
                  "structure [xmin, xmax, ymin, ymax, zmin, zmax]")
            return

        # Shrink region extents if they reach beyond the domain
        if region:
            if region[0] < self.xmin:
                region[0] = self.xmin
            if region[1] > self.xmax:
                region[1] = self.xmax
            if region[2] < self.ymin:
                region[2] = self.ymin
            if region[3] > self.ymax:
                region[3] = self.ymax
            if region[4] < self.zmin:
                region[4] = self.zmin
            if region[5] > self.zmax:
                region[5] = self.zmax

        if not region:
            total_volume = (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)
            total_cell_volume = np.sum(self.cell_volumes)
        else:
            total_volume = (region[1] - region[0]) * (region[3] - region[2]) * (region[5] - region[4])
            total_cell_volume = 0
            for cell in range(self.n_cells):
                if self.cell_in_region(cell, region):
                    total_cell_volume += self.cell_volumes[cell]

        porosity = total_cell_volume / total_volume

        return porosity

    def cell_in_region(self, cell_id, region, search_type="center"):
        """Returns True if cell, identified by its cellID, is within the region, which should be a list providing the region's extents [xmin, xmax, ymin, ymax, zmin, zmax].
When a cell is determined to be within the region depends on searchType, if this is set to 'center', the cell is within the region if its center point is,
if it is set to 'contains' then the entire cell needs to be within the region (this is faster),
if it is set to 'overlap', the cell only needs to overlap the region to be regarded as within the region.

PARAMETERS
----------
cell_id : int
    ID of the cell to check.
region : array_like
    Array of the region to check as [xmin, xmax, ymin, ymax, zmin, zmax].
search_type : str
    How to check whether the cell is inside the region ('center', 'overlap' or 'contains').

RETURNS
-------
inRegion : bool
    Whether or not the cell is inside of the region."""

        if search_type == "contains":
            in_region = True
        elif search_type == "overlap" or search_type == "center":
            in_region = False
        else:
            print("Invalid argument: searchType. Valid values: 'center', 'contains', 'overlap'")
            return

        if search_type == "center":
            if not hasattr(self, "cell_centers"):
                self.calc_cell_centers()
            center = self.cell_centers[cell_id]
            if region[0] <= center[0] <= region[1] \
                    and region[2] <= center[1] <= region[3] \
                    and region[4] <= center[2] <= region[5]:
                in_region = True
        else:
            for point in self.cell_points[cell_id]:
                coords = self.points[point]
                if region[0] <= coords[0] <= region[1] \
                        and region[2] <= coords[1] <= region[3] \
                        and region[4] <= coords[2] <= region[5]:
                    if search_type == "overlap":
                        # If one point is entirely within the region, cell is included in calculation of the mean
                        in_region = True
                        break
                elif search_type == "contains":
                    in_region = False
                    break
        return in_region

    def save(self, file):
        """Use the pickle module to save object to file, so the VTK file does not have to be reread."""

        f = open(file, "bw")
        pickle.dump(self, f)
        f.close()
        print("Object saved to {0}".format(file))


def read_vtk(filename):
    """Reads unstructured grid data from VTK file.

PARAMETERS
----------
filename : str
    Relative or absolute path to the VTK file.

RETURNS
-------
data : dict
    Dictionary containing the VTK file's data."""

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    output = reader.GetOutput()

    data = _unst_grid_read(output)

    return data


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
        print("WARNING: array of volumes is longer than array of values, not all volumes will be used")

    weighted_sum = 0
    for i in range(len(values)):
        weighted_sum += values[i] * volumes[i]

    weighted_mean = weighted_sum / np.sum(volumes)

    return weighted_mean


def calc_permeability(por, u_mean, kin_visc, density, dp, dx):
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

    k = -1. * float(por) * float(kin_visc) * float(u_mean) * float(density) * float(dx) / float(dp)

    return k


def post_process(vtk, kin_visc, density, margin=0, epsilon=0.00001):
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
    Permeability calculated for the VTKObject with given margin."""

    if "U_x" not in vtk.cell_data:
        # Write vector components of velocity
        vtk.write_vector_components("U")

    # Define box for which permeability will be determined
    box = [vtk.xmin + margin, vtk.xmax - margin, vtk.ymin, vtk.ymax, vtk.zmin, vtk.zmax]
    # Calculate mean pressures at both ends of the region
    p1 = vtk.calc_mean("p", region=[box[0] - epsilon, box[0] + epsilon, box[2], box[3], box[4], box[5]],
                       volume_weighted=True)
    p2 = vtk.calc_mean("p", region=[box[1] - epsilon, box[1] + epsilon, box[2], box[3], box[4], box[5]],
                       volume_weighted=True)
    # Calculate permeability of the region
    por = vtk.calc_porosity(region=box)
    # Calculate mean velocity in x-direction in the region
    Ux_mean = vtk.calc_mean("U_x", region=box, volume_weighted=True)
    # Calculate permeability from porosity, weighted velocity, kinematic viscosity, density, pressure gradient and
    # region length
    k = calc_permeability(por, Ux_mean, kin_visc, density, p2 - p1, box[1] - box[0])

    return por, k


def create_post_processing_script(case_dir, case_name, script_path, vtk_file, kin_visc, density, margin):
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

    header = create_script_header(1, 1, 2, "allq", "{0}_post".format(case_name))

    modules = "module load anaconda3/2019.03\n\n"

    commands = "python3 {0} {1} {2} {3} {4}".format(script_path, case_dir, vtk_file, kin_visc, density, margin)

    script = open("postprocessing.sh", "w")
    script.write(header)
    script.write(modules)
    script.write(commands)
    script.close()


def analyse_rev(vtk_obj, kin_visc, density, step, margin):
    """Calculate porosities and permeabilities of a VTKObject instance over a range of regions within the domain.
The regions that are analysed are determined by a step and a margin. Starting from a region with volume of 0 at the center of the domain,
every calculation the region is extended in the x- and y-direction by the step size. When the region comes within a margin 
(given as percentage of the entire domain size) the calculation is stopped.

PARAMETERS
----------
vtk_obj : VTKObject
    VTKObject to do REV calculation on.
kin_visc : float, int
    Kinematic viscosity of fluid.
density : float, int
    Density of fluid.
step : float, int
    How much is added to the calculation domain each calculation.
margin : float, int
    Margin to be excluded from calculations on all sides of the domain."""

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
    xmin, xmax, ymin, ymax, zmin, zmax = vtk_obj.xmin, vtk_obj.xmax, vtk_obj.ymin, vtk_obj.ymax, vtk_obj.zmin, vtk_obj.zmax
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
        box = [box[0] - step / 2, box[1] + step / 2, box[2] - step / 2, box[3] + step / 2, zmin, zmax]

        if box[0] > xmin + margin * dx and box[1] < xmax - margin * dx and box[2] > ymin + margin * dy and box[3] < ymax - margin * dy:
            print("Calculating porosity and permeability for box x: {0} -> {1}; y: {2} -> {3}; z: {4} -> {5}".format(box[0], box[1], box[2], box[3], box[4], box[5]))

            # Calculate porosity and mean velocity along x-axis in the current region
            por = vtk_obj.calc_porosity(region=box)
            ux_mean = vtk_obj.calc_mean("U_x", region=box, volume_weighted=True)

            # Calculate the pressure gradient over the region
            # First calculate mean pressure over the xmin plane of the region
            p1 = vtk_obj.calc_mean("p", region=[box[0] - 0.001 * dx, box[0] + 0.001 * dx, box[2], box[3], box[4], box[5]], volume_weighted=True) * density
            # Secondly calculate mean pressure over the xmax plane of the region
            p2 = vtk_obj.calc_mean("p", region=[box[1] - 0.001 * dx, box[1] + 0.001 * dx, box[2], box[3], box[4], box[5]], volume_weighted=True) * density

            k = calc_permeability(por, ux_mean, kin_visc, density, p2 - p1, box[1] - box[0])

            file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(box[1] - box[0], box[3] - box[2], box[5] - box[4], por, p2 - p1, k))
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
        from scipy.interpolate import make_interp_spline, BSpline
    except ImportError:
        print("Failed to import make_inter_spline and/or BSpline from scipy.interpolate, exiting.")
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


if __name__ == "__main__":
    caseDir = sys.argv[1]
    filename = sys.argv[2]
    kin_visc, density = sys.argv[3], sys.argv[4]
    if len(sys.argv) > 5:
        margin = sys.argv[5]
    else:
        margin = 0 
    os.chdir(caseDir)
    vtk = VTKObject(filename, calc_volumes=True)
    por, k = post_process(vtk, kin_visc, density, margin=margin)
    print("Writing output file in {0}".format(os.getcwd()))
    outfile = open("{0}{1}out.dat".format(caseDir, os.sep), "w")
    outfile.write("{0},{1}".format(por, k))
    outfile.close()