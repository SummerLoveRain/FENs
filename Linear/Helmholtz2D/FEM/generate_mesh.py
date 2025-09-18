import gmsh
import pygmsh
from init_config import *
from train_config import *

class BOUNDARY_TYPE:
    DIRICHLET = 'Dirichlet'
    NEUMANN = 'Neumann'
    ROBIN = 'Robin'

# Initialize empty geometry using the build in kernel in GMSH
# geometry = pygmsh.geo.Geometry()
geometry = pygmsh.occ.geometry.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

resolution = 0.001

# Add points with finer resolution on left side
points_dict = {
    "OA": model.add_point((lb[0], lb[1], 0), mesh_size=resolution),
    'OB': model.add_point((ub[0], lb[1], 0), mesh_size=resolution),
    'OC': model.add_point((ub[0], ub[1], 0), mesh_size=resolution),
    'OD': model.add_point((lb[0], ub[1], 0), mesh_size=resolution),
}

# Add lines between all points creating the rectangle
# channel_lines = [
#     model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
# ]

lines_dict = {
    "OA->OB": model.add_line(points_dict["OA"], points_dict["OB"]),
    "OB->OC": model.add_line(points_dict["OB"], points_dict["OC"]),
    "OC->OD": model.add_line(points_dict["OC"], points_dict["OD"]),
    "OD->OA": model.add_line(points_dict["OD"], points_dict["OA"])
}

lines_region = [
    lines_dict["OA->OB"],
    lines_dict["OB->OC"],
    lines_dict["OC->OD"],
    lines_dict["OD->OA"]
]


# Create a line loop and plane surface for meshing
loop_region = model.add_curve_loop(lines_region)
# plane_surface = model.add_plane_surface(channel_loop, holes=[circle.curve_loop])
plane_surface_region = model.add_plane_surface(loop_region)


geometry.boolean_fragments(plane_surface_region, plane_surface_region)

# Call gmsh kernel before add physical entities
model.synchronize()


model.add_physical([lines_dict["OA->OB"], lines_dict["OB->OC"], lines_dict["OC->OD"], lines_dict["OD->OA"]], label=BOUNDARY_TYPE.DIRICHLET)

model.add_physical([plane_surface_region], label="Volume")


# model.add_physical(circle.curve_loop.curves, "Obstacle")


geometry.generate_mesh(dim=2)

gmsh.write(getParentDir() + '/' + getCurDirName() + '/mesh.msh')
gmsh.clear()
geometry.__exit__()