"""stlTools: some basic utilities to triangulate shapes and create STL files from sets of faces."""

from stl import mesh
import numpy as np


def triangulate_circle(center, radius, height, n):
    """Triangulate a cylinder with a center, radius, and height,
simplifying its circumference to n vertices. Returns array of faces.

PARAMETERS
----------
center : array_like
    2 value array containing the x and y-coordinates of the center of a circle.
radius : float, int
    The radius of the circle
height : float, int
    Height of the cylinder.
n : int
    Number of vertices representing the circle's circumference.

RETURNS
-------
faces : array_like
    Array of faces making up the cylinder."""

    t = np.array([i / n * 2 * np.pi for i in range(n + 1)])
    x = np.array([center[0] + radius * np.cos(i) for i in t])
    y = np.array([center[1] + radius * np.sin(i) for i in t])
    vertices_bottom = np.array([(x_i, y_i, 0) for x_i, y_i in zip(x, y)])
    vertices_top = np.array([(x_i, y_i, height) for x_i, y_i in zip(x, y)])

    # Populate faces list with faces, each as a list of three vertices, making sure the right-hand rule is applied
    faces = []
    for i in range(n):
        # Side triangles touching bottom pseudo circle
        side_face_bottom = [
            vertices_bottom[i],
            vertices_bottom[i + 1],
            vertices_top[i + 1],
        ]
        faces.append(side_face_bottom)
        # Side triangles touching top pseudo circle
        side_face_top = [vertices_bottom[i], vertices_top[i + 1], vertices_top[i]]
        faces.append(side_face_top)
        # Bottom triangles
        bottom_face = [
            vertices_bottom[i],
            (center[0], center[1], 0),
            vertices_bottom[i + 1],
        ]
        faces.append(bottom_face)
        # Top triangles
        top_face = [
            vertices_top[i],
            vertices_top[i + 1],
            (center[0], center[1], height),
        ]
        faces.append(top_face)

    faces = np.array(faces)

    return faces


def create_mesh_object(faces):
    """Create a Mesh object from an array of faces.

PARAMETERS
----------
faces : array_like
    Array of faces from which a mesh should be created.

RETURNS
-------
mesh_object : Mesh
    Object representing the created mesh."""

    mesh_object = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j, v in enumerate(f):
            mesh_object.vectors[i][j] = v
    return mesh_object


def write_stl(objects, filename):
    """Write an .stl file from a set of Mesh objects.

PARAMETERS
objects : array_like
    Array of Mesh objects to combine into an .stl file.
filename : str
    Filename of the .stl file."""

    combined = mesh.Mesh(np.concatenate([obj.data for obj in objects]))
    combined.save(filename + ".stl")


if __name__ == "__main__":
    faces1 = triangulate_circle((11, 5), 4.5, 2, 20)
    faces2 = triangulate_circle((2, 3), 3, 1, 100)
    obj1 = create_mesh_object(faces1)
    obj2 = create_mesh_object(faces2)
    write_stl([obj1, obj2], "circleTest")
