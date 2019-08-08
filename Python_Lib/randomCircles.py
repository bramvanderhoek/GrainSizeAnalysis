import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ModuleNotFoundError:
    pass

### Local dependencies
import stlTools
from grainStats import TruncLogNorm, DataDistribution


def calc_porosity(grains, area):
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

    por = 1 - np.sum(np.pi * grains ** 2) / area

    return por


def check_valid_grain_distances(x, y, r, points_x, points_y, points_r, mindist=0):
    """Check whether a grain at position (x, y) and radius r fits at this point based on
    existing grains specified in points_x, points_y and points_r, and an optional minimum distance 'mindist'.
    Returns True if the grain fits, else returns False.
    
    PARAMETERS
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
    no_overlap : bool
    Whether or not the input grain fits given the other provided grains and mindist."""

    distances = np.sqrt((points_x - x) ** 2 + (points_y - y) ** 2)
    distances = distances - points_r - r - mindist
    overlap = distances <= 0
    no_overlap = not overlap.any()
    return no_overlap


def place_grains(radii, x, y, xmin, xmax, ymin, ymax, mindist=0, report=True):
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
    keeper_x : array
        Array of x positions of placed grains.
    keeper_y : array
        Array of y positions of placed grains.
    keeper_r : array
        Array of radii of placed grains.
    double_x : array
        Array of x positions of grains placed along the edge of the model due to periodicity.
    double_y : array
        Array of y positions of grains placed along the edge of the model due to periodicity.
    double_r : array
    Array of raddi of grains placed along the edge of t he model due to periodicity."""

    # Initialize arrays to hold the grains
    keeper_x = np.array([])
    keeper_y = np.array([])
    keeper_r = np.array([])

    # Initialize arrays to hold 'duplicate' grains, grains which have been placed intersecting the edge of the domain
    # because another grain was overlapping the edge on the other side of the domain (these grains should not be taken into
    # account when calculating the porosity, since using the entire area of the original edge-overlapping grain is identical to
    # taking the area inside the domain of both the original and the duplicated grain).
    double_x = np.array([])
    double_y = np.array([])
    double_r = np.array([])

    # Loop until an attempt has been done to place all radii
    i = 0
    while len(keeper_r) < len(radii) and i < len(radii):
        # Track whether any new grains have been placed while looping through all locations (x, y)
        no_new_grains = True
        for j in range(len(x)):
            # Get current point and radius
            this_x = x[j]
            this_y = y[j]
            this_r = radii[i]

            # Make sure grain does not overlap with inlet or outlet
            if this_x - this_r - mindist < xmin or this_x + this_r + mindist > xmax:
                continue

            # Check if grain is at the edge (top or bottom) of the domain
            if this_y - this_r < ymin or this_y + this_r > ymax:
                edge_grain = True
                if this_y - this_r < ymin:
                    # If grain is at the bottom, put a corresponding grain at the top
                    other_y = this_y + (ymax - ymin)
                elif this_y + this_r > ymax:
                    # If grain is at the top, put a corresponding grain at the bottom
                    other_y = this_y - (ymax - ymin)
            else:
                edge_grain = False

            # Make temporary array holding all grains, including duplicates
            all_x = np.append(keeper_x, double_x)
            all_y = np.append(keeper_y, double_y)
            all_r = np.append(keeper_r, double_r)

            # If grain is far away enough from existing grains, keep the grain
            if check_valid_grain_distances(
                this_x, this_y, this_r, all_x, all_y, all_r, mindist
            ):
                if edge_grain and not check_valid_grain_distances(
                    this_x, other_y, this_r, all_x, all_y, all_r, mindist
                ):
                    # Don't place grain if the duplicate grain is too close to other grains
                    continue

                keeper_x = np.append(keeper_x, this_x)
                keeper_y = np.append(keeper_y, this_y)
                keeper_r = np.append(keeper_r, this_r)

                if edge_grain:
                    double_x = np.append(double_x, this_x)
                    double_y = np.append(double_y, other_y)
                    double_r = np.append(double_r, this_r)

                if report:
                    print(
                        "Placed grain {0}/{1} @ point #{2}{3}".format(
                            i + 1, len(radii), j, " (edge grain)" if edge_grain else ""
                        )
                    )

                # Grain has been found this loop
                no_new_grains = False
                i += 1

            # Break if all grain sizes have been used/tried
            if len(keeper_r) == len(radii) or i == len(radii):
                break
        # Break if no new grains have been found looping through all locations
        if no_new_grains:
            i += 1

    return keeper_x, keeper_y, keeper_r, double_x, double_y, double_r


def create_model(
    model,
    domain,
    number_of_points=500000,
    stl_filename="output",
    path=".",
    points_per_circle=50,
    plotting=False,
    report=True,
):
    """Create a model with the input grain size distribution (log-normal), and a specified porosity, and write it to a Stereolithography file.

    PARAMETERS
    ----------
    model : dict
        Dictionary containing parameters of the grain size distribution.
    
        Mandatory keywords:
            distribution : dict
                Dictionary representing a distribution, either directly as a distribution,
                or defined by field data. Mandatory keywords depend on distribution type.
            mindist : int, float
                Minimum distance between grains (mm).
            seed : bool, int
                If False, no specific seed will be used, else should be a seed for generating a realization of the grain size distribution.
                
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
        por_min : float
            Minimum porosity value of the model
        por_max : float
            Maximum porosity value of the model
        height : int, float
            Height of the domain (i.e. its thickness along the z-axis) (mm).

    number_of_points : int
        Amount of randomly generated points that will be tried out when trying to place a grain into the model
    stl_filename : str
        Filename of the output .stl file.
    path : str
        Path to which the output (.stl and data) files are written (either relative or absolute).
    points_per_circle : int
        Number of points representing a single circle in the output .stl file (more points give a more accurate representation).
    plotting : bool
        Whether or not to plot output and intermediate steps.
    report : bool
        Whether or not to print info to the screen.
    
    RETURNS
    -------
    por_final : float
        The final porosity of the model."""

    # Get domain extents from dictionary
    xmin = domain["xmin"]
    xmax = domain["xmax"]
    ymin = domain["ymin"]
    ymax = domain["ymax"]
    stl_height = domain["cell_size"]

    por_min = domain["por_min"]
    por_max = domain["por_max"]

    # Get distribution statistics from dictionary
    # NOTE: when different distributions are available, the parameters will be different and the entire dictionary should probably be passed to a function!
    #    dist = model["distribution"]

    #    dist = model["distribution"]
    mindist = model["mindist"]
    seed = model["seed"]

    # Generate random points
    if seed is not False:
        np.random.seed(seed)

    # Generate random porosity within the specified range, and initialize random points to place the grains at
    por = np.random.rand() * (por_max - por_min) + por_min
    x = np.random.rand(number_of_points) * (xmax - xmin) + xmin
    y = np.random.rand(number_of_points) * (ymax - ymin) + ymin

    # Calculate mesh area
    mesh_area = (xmax - xmin) * (ymax - ymin)

    # Check which type of distribution to use (only one at the moment).
    if model["model_type"].lower() in ["trunc_lognormal", "trunclognormal"]:
        rmin = model["model_data"]["rmin"]
        rmax = model["model_data"]["rmax"]
        rmean = model["model_data"]["rmean"]
        rstd = model["model_data"]["rstd"]
        # Generate grains based on truncated log normal distribution
        pdf_gen = TruncLogNorm(rmin, rmax, loc=rmean, scale=rstd)
    elif model["model_type"].lower() in ["data_distribution", "data"]:
        data_points = model["model_data"]["data_points"]
        c_freq = model["model_data"]["c_freq"]
        pdf_gen = DataDistribution(data_points, c_freq)
    else:
        raise ValueError("Model tpe not properly defined")

    r = np.array([])
    por_new = 1
    # Add new grains, pulled from the specified distribution, until the right porosity is (approximately) reached.
    # Note: actual porosity will always be slightly lower than the randomly generated one
    while por_new > por:
        r = np.append(r, pdf_gen.rvs(1))
        por_new = calc_porosity(r, mesh_area)
    # Remove grains if porosity has gotten below the minimum specified porosity.
    while por_new < por_min:
        r = r[:-1]
        por_new = calc_porosity(r, mesh_area)

    # Get total number of grains
    number_r = len(r)

    if plotting:
        fig, ax = plt.subplots()
        ax.hist(r, bins=number_r // 5, density=True)
        pdf_x = np.arange(rmin, rmax, 0.01)
        pdf_y = pdf_gen.pdf(pdf_x)
        ax.plot(pdf_x, pdf_y, label="pdf")
        plt.xlabel("r [mm]")
        plt.ylabel("density")
        plt.legend()
        plt.show()

    # Sort grains from large to small
    r.sort()
    r = r[::-1]

    # Place grains into the model domain
    keeper_x, keeper_y, keeper_r, double_x, double_y, double_r = place_grains(
        r, x, y, xmin, xmax, ymin, ymax, mindist, report=report
    )

    # Report if not all grains were placed, and show new distribution of grains sizes
    if len(keeper_r) < number_r:
        print(
            "WARNING: Not all grains were placed into the domain; specified number: {0}, new number: {1}".format(
                number_r, len(keeper_r)
            )
        )
        if plotting:
            pdf_x = np.arange(rmin, rmax, 0.01)
            pdf_y = pdf_gen.pdf(pdf_x)
            fig, ax = plt.subplots()
            ax.hist(keeper_r, bins=len(r) // 5, density=True)
            ax.plot(pdf_x, pdf_y, label="pdf")
            plt.legend()
            plt.show()

    # Calculate final mean grain size and porosity
    # Even though some of the 'keeper' grains are partly outside of the domain,
    # the porosity and mean radius of the domain is still defined as the total area of the 'keeper' grain divided by the mesh area,
    # since all grains at the edge have a corresponding grain at the other side of the model, saved as the 'double' grains.
    rmean_final = np.mean(keeper_r)
    por_final = calc_porosity(keeper_r, mesh_area)

    if report:
        print(
            "Starting porosity: {0}\nPorosity before placing grains: {1}\nFinal porosity: {2}\nFinal mean: {3}".format(
                por, por_new, por_final, rmean_final
            )
        )

    por_file = open("{0}{1}porosity.dat".format(path, os.sep), "w")
    por_file.write(str(por_final) + "\n")
    por_file.close()

    if plotting:
        try:
            fig, ax = plt.subplots()
            for i in range(len(keeper_r)):
                c = Circle((keeper_x[i], keeper_y[i]), keeper_r[i])
                ax.add_patch(c)
            for i in range(len(double_r)):
                c = Circle((double_x[i], double_y[i]), double_r[i])
                ax.add_patch(c)
            ax.autoscale_view()
            plt.show()
        except:
            pass

    # Find a point inside the mesh that is half of mindist away from a grain, to ensure the point is not inside a grain
    # This will be used by OpenFOAM's snappyHexMesh tool to see which part of the geometry is outside of the grains.

    for i in range(len(keeper_r)):
        point_x, point_y = keeper_x[i], keeper_y[i]
        if xmax > point_x > xmin and ymax > point_y > ymin:
            if point_y + keeper_r[i] < ymax:
                point_y += keeper_r[i] + mindist / 2
            elif point_y - keeper_r[i] > ymin:
                point_y -= keeper_r[i] + mindist / 2
            # Check with grains duplicated for periodicity too
            if check_valid_grain_distances(
                point_x, point_y, 0, double_x, double_y, double_r
            ):
                break

    location_file = open("{0}{1}locationInMesh.dat".format(path, os.sep), "w")
    location_file.write("{0} {1} {2}".format(point_x, point_y, stl_height / 2))
    location_file.close()

    # Write raw x, y and radius data to file
    raw_data_file = open("{0}{1}raw_data.dat".format(path, os.sep), "w")
    raw_data_file.write("x,y,r\n")
    for i in range(len(keeper_r)):
        raw_data_file.write(
            "{0},{1},{2}\n".format(keeper_x[i], keeper_y[i], keeper_r[i])
        )
    for i in range(len(double_r)):
        raw_data_file.write(
            "{0},{1},{2}\n".format(double_x[i], double_y[i], double_r[i])
        )
    raw_data_file.close()

    # Given a number of vertices (pointsPerCirle) and a height (stl_height), create an .stl file containing triangulated pseudo-cylinders
    # from the circles in the model.
    mesh_objects = []
    for i in range(len(keeper_r)):
        faces = stlTools.triangulate_circle(
            (keeper_x[i], keeper_y[i]), keeper_r[i], stl_height, points_per_circle
        )
        mesh_object = stlTools.create_mesh_object(faces)
        mesh_objects.append(mesh_object)
    for i in range(len(double_r)):
        faces = stlTools.triangulate_circle(
            (double_x[i], double_y[i]), double_r[i], stl_height, points_per_circle
        )
        mesh_object = stlTools.create_mesh_object(faces)
        mesh_objects.append(mesh_object)
    stlTools.write_stl(mesh_objects, "{0}{1}{2}".format(path, os.sep, stl_filename))

    return por_final


if __name__ == "__main__":
    # Grain size statistics
    model = dict(
        distribution_type="truncLogNormal",
        rmin=0.05,
        rmax=0.8,
        rmean=0.25,
        rstd=0.64,
        mindist=0.025,
        seed=False,
    )

    domain = dict(xmin=0, xmax=10, ymin=0, ymax=10, por_min=0.3, por_max=0.4, height=1)

    n = 10
    por = []
    for i in range(n):
        por.append(create_model(model, domain, plotting=False))
    print(por)
    print(np.mean(por))
