###############################################################################
# The following is a python script that uses VTK to generate fast
# projections of STL meshes for frontal area calculations.  See README for more
# information.
###############################################################################
import argparse
import os
import time
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from xvfbwrapper import Xvfb

from ppcfd.script.starccm_plus.read_case import read_case


def valid_file(param):
    base, ext = os.path.splitext(param)
    if ext.lower() not in (".stl", ".case"):
        raise argparse.ArgumentTypeError("\n\nERROR: File must be an STL mesh.\n")
    return param


def unstructured_to_poly(u_data):
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(u_data)
    geometryFilter.Update()
    polydata_structed = geometryFilter.GetOutput()
    return polydata_structed


def scale_geo(p_data, scale_times=1000):
    transform = vtk.vtkTransform()
    transform.Scale(scale_times, scale_times, scale_times)  # 放大1000倍
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(p_data)
    transformFilter.Update()
    p_data = transformFilter.GetOutput()
    return p_data


def get_bounds(p_data, x_bounds, y_bounds, z_bounds):
    bounds = p_data.GetBounds()
    x_bounds = [min(bounds[0], x_bounds[0]), max(bounds[1], x_bounds[1])]
    y_bounds = [min(bounds[2], y_bounds[0]), max(bounds[3], y_bounds[1])]
    z_bounds = [min(bounds[4], z_bounds[0]), max(bounds[5], z_bounds[1])]
    return x_bounds, y_bounds, z_bounds


def calculate_frontal_area(
    file_name,
    proj_axis,
    ground=-10000,
    max_frame=1500,
    fitfactor=1.0,
    debug=False,
    vtk_data=None,
):
    vdisplay = Xvfb()
    vdisplay.start()
    if debug:
        dprint = print
    else:

        def debug_print(x, *ex):
            pass  # else ignore them

        dprint = debug_print

    # fit factor is a scaling to help fit to scale inside view window.
    # usually needs to be within 1.0 to 2.0 for cars <-> trucks

    dprint("---------------------------------------------------------------------")

    dprint("FITFACTOR = ", fitfactor)
    # ------------------ VTK Render Setup -----------------------------------------
    # max_frame = args.res  # sets max resolution, render windows size

    # Create the renderer
    ren = vtk.vtkRenderer()

    # Create the window
    renWin = vtk.vtkRenderWindow()

    # Add the renderer to the window
    renWin.AddRenderer(ren)

    # Define an interactor.
    iren = vtk.vtkRenderWindowInteractor()

    # Set the window defined above as the window that the interactor
    # will work on.
    iren.SetRenderWindow(renWin)
    x_bounds = [0.0, 0.0]
    y_bounds = [0.0, 0.0]
    z_bounds = [0.0, 0.0]
    if isinstance(vtk_data, vtk.vtkPolyData):
        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(vtk_data)
        geometryFilter.Update()
        polydata_structed = geometryFilter.GetOutput()
        car_surface_data = polydata_structed
        car_surface_data = scale_geo(car_surface_data, 1000)
        x_bounds, y_bounds, z_bounds = get_bounds(
            car_surface_data, x_bounds, y_bounds, z_bounds
        )
        surfaceMapper = vtk.vtkPolyDataMapper()
        surfaceMapper.SetInputData(car_surface_data)
    elif isinstance(vtk_data, vtk.vtkMultiBlockDataSet):
        multi_blockdata = vtk_data
        for i in range(multi_blockdata.GetNumberOfBlocks()):
            t0 = time.time()
            u_data = multi_blockdata.GetBlock(i)
            p_data = unstructured_to_poly(u_data)
            # 创建一个变换过滤器，应用变换
            p_data = scale_geo(p_data, 1000)
            x_bounds, y_bounds, z_bounds = get_bounds(
                p_data, x_bounds, y_bounds, z_bounds
            )

            multi_blockdata.SetBlock(i, p_data)
        car_surface_data = multi_blockdata
        surfaceMapper = vtk.vtkCompositePolyDataMapper()
        surfaceMapper.SetInputDataObject(car_surface_data)
    elif Path(file_name).suffix == ".stl":
        # Initialize the object that is used to load the .stl file
        stlFileReader = vtk.vtkSTLReader()

        # Specify the .stl file's name.
        stlFileReader.SetFileName(file_name)

        # Load the .stl file.
        stlFileReader.Update()
        car_surface_data = stlFileReader.GetOutput()
        car_surface_data = scale_geo(car_surface_data, 1000)
        x_bounds, y_bounds, z_bounds = get_bounds(
            car_surface_data, x_bounds, y_bounds, z_bounds
        )
        # Clipping plane for ground
        plane = vtk.vtkPlane()
        plane.SetNormal(0, 0, 1)
        plane.SetOrigin(0, 0, ground)

        clip = vtk.vtkClipPolyData()
        clip.SetClipFunction(plane)
        clip.SetInputData(car_surface_data)
        clip.Update()
        car_surface_data = clip.GetOutput(0)
        surfaceMapper = vtk.vtkPolyDataMapper()
        surfaceMapper.SetInputData(car_surface_data)
    elif Path(file_name).suffix == ".case":
        _, vtk_data, _ = read_case(file_name)
        multi_blockdata = vtk_data
        for i in range(multi_blockdata.GetNumberOfBlocks()):
            u_data = multi_blockdata.GetBlock(i)
            p_data = unstructured_to_poly(u_data)
            p_data = scale_geo(p_data, 1000)
            x_bounds, y_bounds, z_bounds = get_bounds(
                p_data, x_bounds, y_bounds, z_bounds
            )

            multi_blockdata.SetBlock(i, p_data)
        car_surface_data = multi_blockdata
        surfaceMapper = vtk.vtkCompositePolyDataMapper()
        surfaceMapper.SetInputDataObject(car_surface_data)
    else:
        raise NotImplementedError

    bounds = x_bounds + y_bounds + z_bounds

    dprint("Bounds in x = ", x_bounds)
    dprint("Bounds in y = ", y_bounds)
    dprint("Bounds in z = ", z_bounds)

    intXdim = int(abs(x_bounds[0]) + abs(x_bounds[1]))
    intYdim = int(abs(y_bounds[0]) + abs(y_bounds[1]))
    intZdim = int(abs(z_bounds[0]) + abs(z_bounds[1]))
    dprint("model X Dimension = ", intXdim)
    dprint("model Y Dimension = ", intYdim)
    dprint("model Z Dimension = ", intZdim)

    # Check for model dimensions that aren't going to work.
    if (intXdim or intYdim or intZdim) < 100:
        dprint("Model dimensions small, methods likely innacurate.")
        dprint("Program meant for vehicle models in mm units.")
        dprint("Program exiting...")
        quit()

    if (intXdim or intYdim or intZdim) > 30000:
        dprint("Model dimensions very large, methods likely inoperable.")
        dprint("Program exiting...")
        quit()

    if proj_axis == "X":
        frame_width = intYdim
        frame_height = intZdim

    elif proj_axis == "Y":
        frame_width = intXdim
        frame_height = intZdim

    elif proj_axis == "Z":
        frame_width = intXdim
        frame_height = intYdim
    else:
        raise ValueError("proj_axis must be one of [X, Y, Z]")

    minscale = min([frame_width, frame_height])
    maxscale = max([frame_width, frame_height])

    aspect = frame_width / frame_height

    dprint("ASPECT RATIO = ", aspect)
    dprint("FRAME WIDTH = ", frame_width)
    dprint("FRAME HEIGHT = ", frame_height)

    # Find center
    x_center = int(np.average(x_bounds))
    y_center = int(np.average(y_bounds))
    z_center = int(np.average(z_bounds))

    dprint("X CENTER = ", x_center)
    dprint("Y CENTER = ", y_center)
    dprint("Z CENTER = ", z_center)

    # Initialize the actor
    surfaceActor = vtk.vtkActor()
    surfaceActor.SetMapper(surfaceMapper)

    # Render projection all white
    surfaceActor.GetProperty().SetColor(255, 255, 255)

    # Add the actor to the renderer
    ren.AddActor(surfaceActor)

    # Interactor initialize
    iren.Initialize()
    camera = ren.GetActiveCamera()
    camera.ParallelProjectionOn()

    camera.SetParallelScale(max_frame * fitfactor)  # fitfactor used here

    camera.SetClippingRange(-100000, 100000)  # really large so no clipping

    # Set up view on center of data
    if proj_axis == "X":
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition((x_center + 1), y_center, z_center)

    if proj_axis == "Y":
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(x_center, (y_center + 1), z_center)

    if proj_axis == "Z":
        camera.SetViewUp(0, 1, 0)
        camera.SetPosition(x_center, y_center, (z_center + 1))

    camera.SetFocalPoint(x_center, y_center, z_center)

    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(renWin)
    renWin.SetSize(int(max_frame * aspect), max_frame)
    grabber.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetInputData(grabber.GetOutput())
    writer.SetFileName(os.path.splitext(file_name)[0] + "_output.png")
    writer.Write()

    img = grabber.GetOutput()
    rows, cols, _ = img.GetDimensions()
    img = vtk_to_numpy(img.GetPointData().GetScalars())
    img = img.reshape(cols, rows, -1)
    img = (np.dot(img[..., :3], [1 / 3, 1 / 3, 1 / 3])).astype(int)  # rgb to mono

    borders = [img[0, :], img[-1, :], img[:, 0], img[:, -1]]
    borderCount = 0
    for elem in borders:
        borderCount += np.sum(elem)

    if borderCount > 0:  # white pixels on border means model got cropped!
        dprint("\nModel is not fit to render window, results cannot be computed")
        dprint("accurately.  Change argument -fitfactor to a greater value.")
        dprint("Program exiting...")
        quit()

    # ------------------ Area Calculation -----------------------------------------
    # Take the output and do the area calculation
    n_white_pixels = np.sum(img != 0)  # sum the white pixels in the render
    proj_area = (n_white_pixels * (fitfactor**2)) / (250000)  # math for px/m^2

    # Print the details to the command line

    dprint("\nFRONTAL AREA PROJECTION")
    dprint("----------------------------------------------------------------------")
    dprint("Assumes STL in millimeters, area will be calculated in square meters.")
    dprint("Projected Direction is in: \t\t", proj_axis, " axis\n")
    dprint("Projected Area: \t\t\t", proj_area, " m^2\n")
    dprint("Projected Area in StarCCM+: \t\t", 2.195138, " m^2\n")
    dprint("Non Black Pixels: \t\t\t", n_white_pixels, "\n")
    dprint("Time to Compute: \t\t\t", time.process_time(), " seconds")
    dprint("----------------------------------------------------------------------")
    vdisplay.stop()
    return proj_area, bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=valid_file, help="Filename of STL mesh")
    parser.add_argument(
        "-fitfactor", help="Factor to help fit view to model", type=float, default=1.0
    )
    parser.add_argument(
        "-res", help="Render Window Resolution (default 1500px)", type=int, default=1500
    )
    parser.add_argument(
        "-ground",
        type=int,
        default=-10000,
        help="Ground height clipping in mm from Z=0",
    )
    args = parser.parse_args()
    frontal_area, bounds = calculate_frontal_area(
        args.file_name, "X", args.ground, args.res, args.fitfactor, True
    )
