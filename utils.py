import dolfinx.plot as plot
import pyvista

def plot_density(rho,outputFile=None):
    """
    Plot the displacement and damage field with pyvista
    """
    mesh = rho.function_space.mesh

    topology, cell_types, geometry = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[rho.name] = rho.x.array
    grid.set_active_scalars(rho.name)

    plotter = pyvista.Plotter(title="density", window_size=[800, 300],shape=(1, 1))
    plotter.subplot(0, 0)
    plotter.add_text("Scalar contour field", font_size=1, position="upper_edge")
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=False)
    plotter.view_xy()

    if pyvista.OFF_SCREEN or outputFile:
        plotter.screenshot(
            filename=outputFile,
            transparent_background=False,
            window_size=[800, 800],
        )
    else:
        plotter.show()
    plotter.close()