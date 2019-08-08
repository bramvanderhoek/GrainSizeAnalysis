"""vtkTools.py: selected tools for dealing with .vtk files taken from the ogs5py Python package by Sebastian Müller (https://github.com/GeoStat-Framework/ogs5py)."""

import numpy as np
import os
import pickle

from scipy.spatial import ConvexHull
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
from vtk import vtkUnstructuredGridReader

class VTKObject:
    """Represents the data inside a VTK file, includes methods to do post-processing with the data."""

    def __init__(self, file_path, calc_volumes=False):
        # Attempt to read VTK file from filepath

        self.file_path=file_path
        if not os.path.isfile(self.file_path):
            print("ERROR: {0} is not a file".format(self.file_path))
        elif not file_path.lower().endswith(".vtk"):
            print(
                "ERROR: {0} does not have the proper file extension (.vtk)".format(
                    file_path
                )
            )
        else:
            print("Opening input file {0}".format(self.file_path))
            # Read in data from vtk file
            data = self.read_vtk()
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
            self.cell_data = dict(
                [
                    [
                        attr,
                        np.ndarray(self.n_cells, dtype=type(cell_data[keys[0]][attr])),
                    ]
                    for attr in attrs
                ]
            )
            for i in range(self.n_cells):
                for key in keys:
                    if (
                        len(cell_data[key]["cellID"]) > 0
                        and cell_data[key]["cellID"][0] == i
                    ):
                        self.cell_points[i], cells[key] = cells[key][0], cells[key][1:]
                        for attr in attrs:
                            self.cell_data[attr][i], cell_data[key][attr] = (
                                cell_data[key][attr][0],
                                cell_data[key][attr][1:],
                            )
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

    def read_vtk(self):
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
        reader.SetFileName(self.file_path)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
    
        output = reader.GetOutput()
    
        data = _unst_grid_read(output)
    
        return data


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
            center = np.array(
                [np.mean([xmax, xmin]), np.mean([ymax, ymin]), np.mean([zmax, zmin])]
            )
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
                Mean value calculated from the data.
        """
            
        # Check if region argument has proper length, otherwise raise error.
        if region and not len(region) == 6:
            print(
                "ERROR: provided argument 'region' does not have length of 6. 'region' should be a list with structure [xmin, xmax, ymin, ymax, zmin, zmax]"
            )
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
            """ TODO: adapt change in code """
        elif volume_weighted:
            mean = 0
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
            print(
                "ERROR: {0} was not found as a {1} data attribute".format(
                    vector_attribute, data_type
                )
            )
            return
        if not isinstance(data_dict[vector_attribute][0], np.ndarray):
            print(
                "ERROR: {0} in {1} data is not vector data".format(
                    vector_attribute, data_type
                )
            )
            return
        if not len(data_dict[vector_attribute][0]) == 3:
            print(
                "ERROR: {0} {1} data are not 3-dimensional vectors".format(
                    vector_attribute, data_type
                )
            )
            return

        print(
            "Writing x-, y- and z-components of {0} for {1} {2}{3}".format(
                vector_attribute, n, data_type, "s" if n > 1 else ""
            )
        )

        # Initialize data arrays
        data_dict["{0}_x".format(vector_attribute)] = np.ndarray(n)
        data_dict["{0}_y".format(vector_attribute)] = np.ndarray(n)
        data_dict["{0}_z".format(vector_attribute)] = np.ndarray(n)

        # Split vectors up into components
        for i in range(n):
            data_dict["{0}_x".format(vector_attribute)][i] = data_dict[
                vector_attribute
            ][i][0]
            data_dict["{0}_y".format(vector_attribute)][i] = data_dict[
                vector_attribute
            ][i][1]
            data_dict["{0}_z".format(vector_attribute)][i] = data_dict[
                vector_attribute
            ][i][2]

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
            print(
                "ERROR: provided argument 'region' does not have length of 6. 'region' should be a list with "
                "structure [xmin, xmax, ymin, ymax, zmin, zmax]"
            )
            return

        # Shrink region extents if they reach beyond the domain
        if region:
            region[0] = np.max([region[0], self.xmin])
            region[1] = np.min([region[1], self.xmax])
            region[2] = np.max([region[2], self.ymin])
            region[3] = np.min([region[3], self.ymax])
            region[4] = np.max([region[4], self.zmin])
            region[5] = np.min([region[5], self.zmax])

        if not region:
            total_volume = (
                (self.xmax - self.xmin)
                * (self.ymax - self.ymin)
                * (self.zmax - self.zmin)
            )
            total_cell_volume = np.sum(self.cell_volumes)
        else:
            total_volume = (
                (region[1] - region[0])
                * (region[3] - region[2])
                * (region[5] - region[4])
            )
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
            print(
                "Invalid argument: searchType. Valid values: 'center', 'contains', 'overlap'"
            )
            return

        if search_type == "center":
            if not hasattr(self, "cell_centers"):
                self.calc_cell_centers()
            center = self.cell_centers[cell_id]
            if (
                region[0] <= center[0] <= region[1]
                and region[2] <= center[1] <= region[3]
                and region[4] <= center[2] <= region[5]
            ):
                in_region = True
        else:
            for point in self.cell_points[cell_id]:
                coords = self.points[point]
                if (
                    region[0] <= coords[0] <= region[1]
                    and region[2] <= coords[1] <= region[3]
                    and region[4] <= coords[2] <= region[5]
                ):
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

###############################################################################
### Additional Functionalities    
###############################################################################

# coresponding vtk-types by their number encoding
VTK_TYP = {
    3: "line",  # vtk.VTK_LINE == 3
    5: "tri",  # vtk.VTK_TRIANGLE == 5
    9: "quad",  # vtk.VTK_QUAD == 9
    10: "tet",  # vtk.VTK_TETRA == 10
    14: "pyra",  # vtk.VTK_PYRAMID == 14
    13: "pris",  # vtk.VTK_WEDGE == 13
    12: "hex",  # vtk.VTK_HEXAHEDRON == 12
    "line": 3,
    "tri": 5,
    "quad": 9,
    "tet": 10,
    "pyra": 14,
    "pris": 13,
    "hex": 12,
}
"""dict: vtk type codes per element name"""

# number of nodes per element-type (sorted by name and number-encoding)
NODE_NO = {
    0: 2,
    1: 3,
    2: 4,
    3: 4,
    4: 5,
    5: 6,
    6: 8,
    "line": 2,
    "tri": 3,
    "quad": 4,
    "tet": 4,
    "pyra": 5,
    "pris": 6,
    "hex": 8,
}
"""dict: Node numbers per element name"""

def _get_data(data):
    """
    extract data as numpy arrays from a vtkObject
    """
    arr_dict = {}
    no_of_arr = data.GetNumberOfArrays()
    for i in range(no_of_arr):
        arr = data.GetArray(i)
        if arr:
            arr_dict[arr.GetName()] = vtk2np(arr)
    return arr_dict

def _get_cells(obj):
    """
    extract cells and cell_data from a vtkDataSet
    and sort it by cell types
    """
    cells, cell_data = {}, {}
    data = _get_data(obj.GetCellData())
    arr = vtk2np(obj.GetCells().GetData())
    loc = vtk2np(obj.GetCellLocationsArray())
    types = vtk2np(obj.GetCellTypesArray())

    for typ in VTK_TYP:
        if not isinstance(typ, int):
            continue
        cell_name = VTK_TYP[typ]
        n_no = NODE_NO[cell_name]
        cell_loc_i = np.where(types == typ)[0]
        loc_i = loc[cell_loc_i]
        # if there are no cells of the actual type continue
        if len(loc_i) == 0:
            # if not loc_i:
            continue
        arr_i = np.empty((len(loc_i), n_no), dtype=int)
        for i in range(n_no):
            arr_i[:, i] = arr[loc_i + i + 1]
        cells[cell_name] = arr_i
        cell_data_i = {}
        for data_i in data:
            cell_data_i[data_i] = data[data_i][cell_loc_i]
        if cell_data_i != {}:
            cell_data[cell_name] = cell_data_i

    return cells, cell_data

def _unst_grid_read(obj):
    """
    a reader for vtk unstructured grid objects
    """
    output = {}
    # output["field_data"] = _get_data(obj.GetFieldData())
    output["points"] = vtk2np(obj.GetPoints().GetData())
    output["point_data"] = _get_data(obj.GetPointData())
    output["cells"], output["cell_data"] = _get_cells(obj)
    return output