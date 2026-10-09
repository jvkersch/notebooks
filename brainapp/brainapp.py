import nibabel as ni

from mayavi import mlab
from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi.sources.api import VTKDataSource

# Standard imports.
from numpy import sqrt, sin, mgrid

# Enthought imports.
from traits.api import HasStrictTraits, Instance, Property, Enum, Array, on_trait_change, Float, Range, Int, Any, Bool
from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup, VGroup, UItem, Label, RangeEditor
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.engine_view import EngineView
from mayavi.tools.mlab_scene_model import MlabSceneModel

from mayavi.core.api import PipelineBase, Source
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Outline, Surface, IsoSurface, ScalarCutPlane


class BrainApp(HasStrictTraits):

    data = Array()

    module_3d = Any

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # visualization style
    show_cut_planes = Bool(True)
    show_isosurface = Bool(True)

    data_min = Float
    data_max = Float
    contour = Float
    opacity = Float(0.5)

    _axis_names = dict(x=0, y=1, z=2)

    def default_traits_view(self):
        view = View(
            HSplit(
                HGroup(
                    VGroup(
                        UItem('scene_y',
                              editor=SceneEditor(scene_class=Scene)),
                        UItem('scene_z',
                              editor=SceneEditor(scene_class=Scene)),
                        UItem('scene_x',
                              editor=SceneEditor(scene_class=Scene)),
                        columns=2,
                    ),
                    UItem(name='scene3d',
                          editor=SceneEditor(),
                          resizable=True,
                          height=500,
                          width=500),
                ),
                VGroup(
                    Item("show_cut_planes"),
                    Item("show_isosurface"),
                    Item("contour",
                         enabled_when="show_isosurface",
                         editor=RangeEditor(
                             format="%.02f",
                             low_name="data_min",
                             high_name="data_max",
                             mode="slider")),
                    Item("opacity",
                         editor=RangeEditor(
                             format="%.02f",
                             low=0.0,
                             high=1.0,
                             mode="slider")),
                ),
            ),
            resizable=True,
            scrollable=True
        )
        return view

    def _data_min_default(self):
        return self.data.min()

    def _data_max_default(self):
        return self.data.max()

    def _contour_default(self):
        return (self.data_min + self.data_max)/2

    def __init__(self, **traits):
        super().__init__(**traits)
        self.ipw_3d_x = self.make_ipw_3d("x")
        self.ipw_3d_y = self.make_ipw_3d("y")
        self.ipw_3d_z = self.make_ipw_3d("z")


    def _data_default(self):
        ni_file = ni.load("BoMSubject09.nii")
        return ni_file.get_fdata()

    def _data_src3d_default(self):
        return mlab.pipeline.scalar_field(self.data,
                                          figure=self.scene3d.mayavi_scene)

    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        print("scene3d")
        outline = mlab.pipeline.outline(self.data_src3d,
                                        figure=self.scene3d.mayavi_scene,
                                        colormap='bone')

        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            # Turn the interaction off
            ipw.ipw.interaction = 0
        # self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()

        # The contour surface needs to sit on its own data source.
        isodata_src3d = mlab.pipeline.scalar_field(
            self.data, figure=self.scene3d.mayavi_scene)

        self.module_3d = surface = mlab.pipeline.iso_surface(
            isodata_src3d, opacity=self.opacity,
        )

        surface.contour.auto_contours = False
        surface.contour.contours = [self.contour]

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(
            self.data_src3d,
            figure=self.scene3d.mayavi_scene,
            plane_orientation='%s_axes' % axis_name,
            colormap='bone'
        )
        return ipw

    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
            self.data_src3d.mlab_source.dataset,
            figure=scene.mayavi_scene,
            colormap='bone'
        )
        ipw = mlab.pipeline.image_plane_widget(
                            outline,
                            plane_orientation='%s_axes' % axis_name,
            colormap='bone')
        # setattr(self, 'ipw_3d_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0
        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            position = obj.GetCurrentCursorPosition()
            for other_axis, axis_number in self._axis_names.items():
                if other_axis == axis_name:
                    continue
                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
                ipw3d.ipw.slice_position = position[axis_number]

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                    self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=( 0, 90),
                     y=(90, 90),
                     z=( 0,  0),
                     )
        scene.mlab.view(*views[axis_name])
        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
            tvtk.InteractorStyleImage()
        # scene.scene.background = (0, 0, 0)

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        print("make side view x")
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        print("make side view y")
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        print("make side view z")
        return self.make_side_view('z')

    @on_trait_change("contour")
    def update_contour(self):
        self.module_3d.contour.contours = [self.contour]

    @on_trait_change("opacity")
    def update_opacity(self):
        self.module_3d.actor.property.opacity = self.opacity

    @on_trait_change("show_cut_planes")
    def update_cut_plane_visibility(self):
        self.ipw_3d_x.visible = self.show_cut_planes
        self.ipw_3d_y.visible = self.show_cut_planes
        self.ipw_3d_z.visible = self.show_cut_planes

    @on_trait_change("show_isosurface")
    def update_iso_vis(self):
        self.module_3d.visible = self.show_isosurface


if __name__ == '__main__':
    m = BrainApp()
    m.configure_traits()