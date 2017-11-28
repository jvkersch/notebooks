""" A simple Traits-UI/Chaco application to interact with t-SNE.

"""

from collections import OrderedDict
import time

from enable.api import Component, ComponentEditor
from traits.api import (
    Array, Button, Enum, HasStrictTraits, Instance, Int, List, Str
)
from traitsui.api import (
     HGroup, Item, RangeEditor, UItem, VGroup, View, spring
)
from chaco.api import ArrayPlotData, ColorMapper, Plot

from sklearn import datasets, manifold
from seaborn import color_palette


CLUSTER_FACTORIES = OrderedDict([
    ('Circles', datasets.make_circles),
    ('S-curves', datasets.make_s_curve),
    ('Gaussian Quantiles', datasets.make_gaussian_quantiles),
    ('Blobs', datasets.make_blobs),
])
CLUSTER_NAMES = list(CLUSTER_FACTORIES.keys())

CLUSTER_COLOR_MAPPER = ColorMapper.from_palette_array(
    color_palette(palette='colorblind', n_colors=5)
)


class Demo(HasStrictTraits):

    current_dataset = Array
    current_labels = Array

    cluster_type = Enum(values="cluster_names")
    cluster_names = List(CLUSTER_NAMES)

    regenerate_button = Button(u"Regenerate Clusters")

    perplexity = Int(10)
    learning_rate = Int(10)
    tsne_runtime_message = Str('')

    view = View(
        VGroup(
            HGroup(
                UItem('cluster_type'),
                UItem('regenerate_button'),
                spring,
                VGroup(
                    Item('perplexity', editor=RangeEditor(low=5, high=100)),
                    Item('learning_rate', editor=RangeEditor(low=10, high=100)),
                ),
            ),
            HGroup(
                UItem('plot', editor=ComponentEditor()),
                UItem('tsne_plot', editor=ComponentEditor()),
            ),
            HGroup(
                spring,
                UItem('tsne_runtime_message', style='readonly'),
            )
        )
    )
    plot = Instance(Component)
    tsne_plot = Instance(Component)
    plot_data = Instance(ArrayPlotData, ())

    def _plot_default(self):
        self._update_data_for_cluster_type()
        plot = Plot(self.plot_data)
        plot.plot(
            ("x", "y", "labels"),
            type="cmap_scatter",
            marker="circle",
            color_mapper=CLUSTER_COLOR_MAPPER,
        )
        return plot

    def _tsne_plot_default(self):
        self._update_tsne_for_perplexity()
        plot = Plot(self.plot_data)
        plot.plot(
            ("tsne_x", "tsne_y", "labels"),
            type="cmap_scatter",
            marker="circle",
            color_mapper=CLUSTER_COLOR_MAPPER,
        )
        return plot

    def _update_tsne_for_perplexity(self):
        tsne = manifold.TSNE(n_components=2, init='random',
                             learning_rate=self.learning_rate,
                             random_state=0, perplexity=self.perplexity)
        start = time.time()
        transformed_xy = tsne.fit_transform(self.current_dataset)
        runtime = time.time() - start
        self.plot_data.update(tsne_x=transformed_xy[:, 0],
                              tsne_y=transformed_xy[:, 1])
        self._update_runtime_message(runtime)

    def _update_data_for_cluster_type(self):
        xy, labels = CLUSTER_FACTORIES[self.cluster_type]()
        self.current_dataset = xy
        self.current_labels = labels
        # Tweak to accommodate S-curves dataset, which is 3d
        self.plot_data.update(x=xy[:, 0], y=xy[:, -1], labels=labels)

    def _cluster_type_changed(self):
        self._update_data_for_cluster_type()
        self._update_tsne_for_perplexity()

    def _regenerate_button_changed(self):
        self._update_data_for_cluster_type()
        self._update_tsne_for_perplexity()

    def _perplexity_changed(self):
        self._update_tsne_for_perplexity()

    def _learning_rate_changed(self):
        self._update_tsne_for_perplexity()

    def _update_runtime_message(self, rt):
        self.tsne_runtime_message = "t-SNE runtime: {:.02f} s.".format(rt)


if __name__ == '__main__':
    Demo().configure_traits()
