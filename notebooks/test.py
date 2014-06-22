# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.display import display
from IPython.html import widgets as W
from IPython.utils import traitlets as T

class CirclesWidget(W.DOMWidget):
    _view_name = T.Unicode('CirclesView', sync=True)
    radii = T.List(sync=True)

# <codecell>

%%javascript
require.config({paths: {d3: "//d3js.org/d3.v3.min"}});

# <codecell>

%%javascript
require(["widgets/js/widget", "d3"], function(WidgetManager, d3){
    var CirclesView = IPython.DOMWidgetView.extend({
        render: function(){
            this.svg = d3.select(this.el).append("svg")
                .attr({
                    width: 500,
                    height: 100
                });

            this.update();
        },
        update: function(){
            var radii = this.model.get("radii"),
                circle = this.svg.selectAll("circle")
                    .data(radii);

            circle.enter().append("circle")
                .style({fill: "red", opacity: .5})
                .attr({cy: 50})
                .on("click", this.click.bind(this));

            circle.transition()
                .attr({
                    cx: function(d, i){ return (i+1) * 50; },
                    r: function(d){ return d * 10; }
                });

            circle.exit()
                .transition()
                .style({fill: "black", opacity: 0})
                .remove();
        },
        click: function(d, i){
            var new_radii = this.model.get("radii").slice();
            new_radii[i] += 1;
            this.model.set("radii", new_radii);
            this.touch();
        }
    });
    WidgetManager.register_widget_view("CirclesView", CirclesView);
});

# <codecell>

circles = CirclesWidget(radii=[1,2,3])
display(circles)

# <codecell>

circles.radii = [3,2,1]

# <codecell>

import time
import random
for i in range(10):
    time.sleep(1)
    circles.radii = random.sample(range(10), random.randint(0, 10))

