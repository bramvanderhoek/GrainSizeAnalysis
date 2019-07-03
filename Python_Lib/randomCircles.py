import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except:
    pass
import stlTools

from grainStats import generateTruncLogNormal

def calcPorosity(grains, area):
    """Calculate porosity given an array of grain sizes and a total surface area.

PARAMETERS
----------
grains : array_like
    Array of grain radii.
area : float, int
    Total area of the domain.

RETURNS
-------
por : float
    The porosity calculated from the grain radii and total area."""

    por = 1 - np.sum(np.pi*grains**2) / area

    return por

def checkValidGrainDistances(x, y, r, points_x, points_y, points_r, mindist=0):
    """Check whether a grain at position (x, y) and radius r fits at this point based on
existing grains specified in points_x, points_y and points_r, and an optional minimum distance 'mindist'.
Returns True if the grain fits, else returns False.

PARAMETES
---------
x : float, int
    x-position of the center of the grain.
y : float, int
    y-position of the center of the grain.
r : float, int
    radius of the grain.
points_x : array_like
    Array of x-coordinates of grains to check distances with.
points_y : array_like
    Array of y-coordinates of grains to check distances with.
points_r : array_like
    Array of radii of grains to check distances with.
mindist : float, int
    Minimum distance allowed between grains

RETURNS
-------
noOverlap : bool
    Whether or not the input grain fits given the other provided grains and mindist."""

    distances = np.sqrt((points_x - x)**2 +  (points_y - y)**2)
    distances = distances - points_r - r - mindist
    overlap = distances <= 0
    noOverlap = not overlap.any()
    return noOverlap

def placeGrains(radii, x, y, xmin, xmax, ymin, ymax, mindist=0, report=True):
    """Place grains by finding a location defined in (x, y) where a grain with radius defined in radii
fits based on previously placed grains and a minimum distance (mindist).

Repeats the loop through all locations if a grain has been placed in the previous loop,
under the assumption that the grains are sorted from large to small, as it gets easier to
place grains the smaller the radius gets.

If no valid location for a single grain was found in the entire loop, skips the grain and continues with the next one.

Implements periodicity: if a grain is placed with its center inside the domain, but overlapping the model edges at the top
or bottom of the domain, a grain is placed just outside the domain at the other side to mimic the model 'looping' around along
the y-axis.

PARAMETERS
----------
radii : array_like
    Array of grain radii of grains to be placed.
x : array_like
    Array of x positions of grains to be placed.
y : array_like
    Array of y positions of grains to be placed.
xmin : int, float
    Lower value of the domain size along the x-axis (mm).
xmax : int, float
    Upper value of the domain size along the x-axis (mm).
ymin : int, float
    Lower value of the domain size along the y-axis (mm).
ymax : int, float
    Upper value of the domain size along the y-axis (mm).
mindist : int, float
    Minimum distance between grains (mm).
report : bool
    Whether or not to print status to stdout.

RETURNS
-------
keeperX : array
    Array of x positions of placed grains.
keeperY : array
    Array of y positions of placed grains.
keeperR : array
    Array of radii of placed grains.
doubleX : array
    Array of x positions of grains placed along the edge of the model due to periodicity.
doubleY : array
    Array of y positions of grains placed along the edge of the model due to periodicity.
doubleR : array
    Array of raddi of grains placed along the edge of t he model due to periodicity."""

    # Initialize arrays to hold the grains
    keeperX = np.array([])
    keeperY = np.array([])
    keeperR = np.array([])

    # Initialize arrays to hold 'duplicate' grains, grains which have been placed intersecting the edge of the domain
    # because another grain was overlapping the edge on the other side of the domain (these grains should not be taken into
    # account when calculating the porosity, since using the entire area of the original edge-overlapping grain is identical to
    # taking the area inside the domain of both the original and the duplicated grain).
    doubleX = np.array([])
    doubleY = np.array([])
    doubleR = np.array([])

    # Loop until an attempt has been done to place all radii
    i = 0
    while len(keeperR) < len(radii) and i < len(radii):
        # Track whether any new grains have been placed while looping through all locations (x, y)
        noNewGrains = True
        for j in range(len(x)):
            # Get current point and radius
            thisX = x[j]
            thisY = y[j]
            thisR = radii[i]

            # Make sure grain does not overlap with inlet or outlet
            if thisX - thisR - mindist < xmin or thisX + thisR  + mindist > xmax:
                continue

            # Check if grain is at the edge (top or bottom) of the domain
            if thisY - thisR < ymin or thisY + thisR > ymax:
                edgeGrain = True
                if thisY - thisR < ymin:
                    # If grain is at the bottom, put a corresponding grain at the top
                    otherY = thisY + (ymax - ymin)
                elif thisY + thisR > ymax:
                    # If grain is at the top, put a corresponding grain at the bottom
                    otherY = thisY - (ymax - ymin)
            else:
                edgeGrain = False

            # Make temporary array holding all grains, including duplicates
            allX = np.append(keeperX, doubleX)
            allY = np.append(keeperY, doubleY)
            allR = np.append(keeperR, doubleR)

            # If grain is far away enough from existing grains, keep the grain
            if checkValidGrainDistances(thisX, thisY, thisR, allX, allY, allR, mindist):
                if edgeGrain and not checkValidGrainDistances(thisX, otherY, thisR, allX, allY, allR, mindist):
                    # Don't place grain if the duplicate grain is too close to other grains
                    continue

                keeperX = np.append(keeperX, thisX)
                keeperY = np.append(keeperY, thisY)
                keeperR = np.append(keeperR, thisR)

                if edgeGrain:
                    doubleX = np.append(doubleX, thisX)
                    doubleY = np.append(doubleY, otherY)
                    doubleR = np.append(doubleR, thisR)

                if report:
                    print("Placed grain {0}/{1} @ point #{2}{3}".format(i+1, len(radii), j, " (edge grain)" if edgeGrain else ""))

                # Grain has been found this loop
                noNewGrains = False
                i += 1

            # Break if all grain sizes have been used/tried
            if len(keeperR) == len(radii) or i == len(radii):
                break
        # Break if no new grains have been found looping through all locations
        if noNewGrains:
            i += 1

    return keeperX, keeperY, keeperR, doubleX, doubleY, doubleR

def createModel(distribution, domain, numberOfPoints=100000, stlFilename="output", path=".", pointsPerCircle=50, plotting=False):
    """Create a model with the input grain size distribution (log-normal), and a specified porosity, and write it to a Stereolithography file.

PARAMETERS
----------
distribution : dict
    Dictionary containing parameters of the grain size distribution.

    Mandatory keywords:
        distributionType : str
            String containing the type of distribution to be used.
            Valid values:
                - "truncLogNormal"
        rmin : int, float
            Minimum grain radius (mm).
        rmax : int, float
            Maximum grain radius (mm).
        rmean : int, float
            Mean grain radius (mm).
        rstd : int, float
            Standard deviation of grain radius (mm).
        mindist : int, float
            Minimum distance between grains (mm).
            
domain : dict
    Dictionary containing parameters of the modelling domain.

    Mandatory keywords:
        xmin : int, float
            Lower value of the domain size along the x-axis (mm).
        xmax : int, float
            Upper value of the domain size along the x-axis (mm).
        ymin : int, float
            Lower value of the domain size along the y-axis (mm).
        ymax : int, float
            Upper value of the domain size along the y-axis (mm).
        por : float
            The porosity of the domain.
        porTolerance : float
            Amount of deviation allowed in the porosity of the model (as decimal percentage of por).
        height : int, float
            Height of the domain (i.e. its thickness along the z-axis) (mm).

numberOfPoints : int
    Amount of randomly generated points that will be tried out when trying to place a grain into the model
stlFilename : str
    Filename of the output .stl file.
path : str
    Path to which the output (.stl and data) files are written (either relative or absolute).
pointsPerCircle : int
    Number of points representing a single circle in the output .stl file (more points give a more accurate representation).
plotting : bool
    Whether or not to plot output and intermediate steps."""
    
    # Get domain extents from dictionary
    xmin = domain["xmin"]
    xmax = domain["xmax"]
    ymin = domain["ymin"]
    ymax = domain["ymax"]

    stlHeight = domain["height"]

    por = domain["por"]
    porTolerance = domain["porTolerance"]

    # Get distribution statistics from dictionary
    # NOTE: when different distributions are available, the parameters will be different and the entire dictionary should probably be passed to a function!
    rmin = distribution["rmin"]
    rmax = distribution["rmax"]
    rmean = distribution["rmean"]
    rstd = distribution["rstd"]
    mindist = distribution["mindist"]

    # Calculate mesh area
    meshArea = (xmax - xmin) * (ymax - ymin)
    
    # Initial estimate of number of spheres needed to reach specified porosity value
    numberR = round(meshArea*(1 - por)/(np.pi*rmean**2))

    # Generate random points
    x = np.random.rand(numberOfPoints) * (xmax - xmin) + xmin
    y = np.random.rand(numberOfPoints) * (ymax - ymin) + ymin

    # Check which type of distribution to use (only one at the moment).
    if distribution["distributionType"] == "truncLogNormal":
        # Generate grains based on truncated log normal distribution
        r = generateTruncLogNormal(numberR, rmin, rmax, rmean, rstd)

    # Calculate theoretical porosity based on generated realization of grain size distribution
    porNew = calcPorosity(r, meshArea)

    # Add or subtract grains until porosity is within acceptable range of wanted porosity
    while not (porNew > por - (por * porTolerance) and porNew < por + (por * porTolerance)):
        if porNew > por:
            numberR += 1
        elif porNew < por:
            numberR -= 1
        r = generateTruncLogNormal(numberR, rmin, rmax, rmean, rstd)
        porNew = calcPorosity(r, meshArea)
        print(porNew)
    
    if plotting:
        try:
            fig, ax = plt.subplots()
            ax.hist(r, bins=numberR//5)
            plt.xlabel("r [mm]")
            plt.ylabel("n")
            plt.show()
        except:
            pass

    # Sort grains from large to small
    r.sort()
    r = r[::-1]

    # Place grains into the model domain
    keeperX, keeperY, keeperR, doubleX, doubleY, doubleR = placeGrains(r, x, y, xmin, xmax, ymin, ymax, mindist)

    # Report if not all grains were placed, and show new distribution of grains sizes
    if len(keeperR) < numberR:
        print("WARNING: Not all grains were placed into the domain; specified number: {0}, new number: {1}".format(numberR, len(keeperR)))
        if plotting:
            try:
                fig, ax = plt.subplots()
                ax.hist(keeperR, bins=len(r)//5)
                plt.show()
            except:
                pass

    # Calculate final mean grain size and porosity
    # Even though some of the 'keeper' grains are partly outside of the domain,
    # the porosity and mean radius of the domain is still defined as the total area of the 'keeper' grain divided by the mesh area,
    # since all grains at the edge have a corresponding grain at the other side of the model, saved as the 'double' grains.
    rmeanFinal = np.mean(keeperR)
    porFinal = calcPorosity(keeperR, meshArea)

    porFile = open("{0}{1}porosity.dat".format(path, os.sep), "w")
    porFile.write(str(porFinal) + "\n")
    porFile.close()

    print("Final mean: {0}".format(rmeanFinal))
    print("Final porosity: {0}".format(porFinal))

    if plotting:
        try:
            fig, ax = plt.subplots()
            for i in range(len(keeperR)):
                c = Circle((keeperX[i], keeperY[i]), keeperR[i])
                ax.add_patch(c)
            for i in range(len(doubleR)):
                c = Circle((doubleX[i], doubleY[i]), doubleR[i])
                ax.add_patch(c)
            ax.autoscale_view()
            plt.show()
        except:
            pass

    # Find a point inside the mesh that is half of mindist away from a grain, to ensure the point is not inside a grain
    # This will be used by OpenFOAM's snappyHexMesh tool to see which part of the geometry is outside of the grains.

    for i in range(len(keeperR)):
        pointX, pointY = keeperX[i], keeperY[i]
        if pointX > xmin and pointX < xmax and pointY > ymin and pointY < ymax:
            if pointY + keeperR[i] < ymax:
                pointY += (keeperR[i] + mindist / 2)
                break
            elif pointY - keeperR[i] > ymin:
                pointY -= (keeperR[i] - mindist / 2)
                break

    locationFile = open("locationInMesh.dat", "w")
    locationFile.write("{0} {1} {2}".format(pointX, pointY, stlHeight / 2))
    locationFile.close()

    # Given a number of vertices (pointsPerCirle) and a height (stlHeight), create an .stl file containing triangulated pseudo-cylinders
    # from the circles in the model.
    meshObjects = []
    for i in range(len(keeperR)):
        faces = stlTools.triangulateCircle((keeperX[i], keeperY[i]), keeperR[i], stlHeight, pointsPerCircle)
        meshObject = stlTools.createMeshObject(faces)
        meshObjects.append(meshObject)
    for i in range(len(doubleR)):
        faces = stlTools.triangulateCircle((doubleX[i], doubleY[i]), doubleR[i], stlHeight, pointsPerCircle)
        meshObject = stlTools.createMeshObject(faces)
        meshObjects.append(meshObject)
    stlTools.writeSTL(meshObjects, "{0}{1}{2}".format(path, os.sep, stlFilename))

if __name__ == "__main__":
    # Grain size statistics
    model = dict(distributionType="truncLogNormal",
            rmin=0.05,
            rmax=0.8,
            rmean=0.35,
            rstd=0.25,
            mindist=0.025)

    domain = dict(xmin=0,
            xmax=12,
            ymin=0,
            ymax=12,
            por=0.35,
            height=1)

    createModel(model, domain, plotting=True)