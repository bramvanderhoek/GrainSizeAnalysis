"""stlTools: some basic utilities to triangulate shapes and create STL files from sets of faces."""

from stl import mesh
import numpy as np

def triangulateCircle(center, radius, height, n):
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

    t = np.array([i/n*2*np.pi for i in range(n+1)])
    x = np.array([center[0] + radius*np.cos(i) for i in t])
    y = np.array([center[1] + radius*np.sin(i) for i in t])
    verticesBottom = np.array([(x_i, y_i, 0) for x_i, y_i in zip(x, y)])
    verticesTop = np.array([(x_i, y_i, height) for x_i, y_i in zip(x, y)])

    # Populate faces list with faces, each as a list of three vertices, making sure the right-hand rule is applied
    faces = []
    for i in range(n):
        # Side triangles touching bottom pseudo circle
        sideFaceBottom = [verticesBottom[i], verticesBottom[i+1], verticesTop[i+1]]
        faces.append(sideFaceBottom)
        # Side triangles touching top pseudo circle
        sideFaceTop = [verticesBottom[i], verticesTop[i+1], verticesTop[i]]
        faces.append(sideFaceTop)
        # Bottom triangles
        bottomFace = [verticesBottom[i], (center[0], center[1], 0), verticesBottom[i+1]]
        faces.append(bottomFace)
        # Top triangles
        topFace = [verticesTop[i], verticesTop[i+1], (center[0], center[1], height)]
        faces.append(topFace)

    faces = np.array(faces)

    return faces

def createMeshObject(faces):
    """Create a Mesh object from an array of faces.

PARAMETERS
----------
faces : array_like
    Array of faces from which a mesh should be created.

RETURNS
-------
meshObject : Mesh
    Object representing the created mesh."""

    meshObject = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j, v in enumerate(f):
            meshObject.vectors[i][j] = v
    return meshObject

def writeSTL(objects, filename):
    """Write an .stl file from a set of Mesh objects.

PARAMETERS
objects : array_like
    Array of Mesh objects to combine into an .stl file.
filename : str
    Filename of the .stl file."""
    
    combined = mesh.Mesh(np.concatenate([obj.data for obj in objects]))
    combined.save(filename + ".stl")

if __name__ == "__main__":
    faces1 = triangulateCircle((11, 5), 4.5, 2, 20)
    faces2 = triangulateCircle((2, 3), 3, 1, 100)
    obj1 = createMeshObject(faces1)
    obj2 = createMeshObject(faces2)
    writeSTL([obj1, obj2], "circleTest")