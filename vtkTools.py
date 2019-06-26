"""vtkTools.py: selected tools for dealing with .vtk files taken from the ogs5py Python package by Sebastian Müller (https://github.com/GeoStat-Framework/ogs5py)."""

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as vtk2np

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