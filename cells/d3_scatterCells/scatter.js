// reference: http://www.cotrino.com/lifespan/lifespan.js

var margin = { top: 50, right: 300, bottom: 50, left: 50 },
outerWidth = 2000,
outerHeight = 1000,
width = outerWidth - margin.left - margin.right,
height = outerHeight - margin.top - margin.bottom;

var x = d3.scale.linear()
.range([0, width]).nice();

var y = d3.scale.linear()
.range([height, 0]).nice();

var xCat = "e1", yCat = "e2";

function create_data(data) {

  var xMax = d3.max(data, function(d) { return d[xCat]; }) * 1.05,
  xMin = d3.min(data, function(d) { return d[xCat]; }),
  xMin = xMin > 0 ? 0 : xMin,
  yMax = d3.max(data, function(d) { return d[yCat]; }) * 1.05,
  yMin = d3.min(data, function(d) { return d[yCat]; }),
  yMin = yMin > 0 ? 0 : yMin;

  x.domain([xMin, xMax]);
  y.domain([yMin, yMax]);

  var xAxis = d3.svg.axis()
  .scale(x)
  .orient("bottom")
  .tickSize(-height);

  var yAxis = d3.svg.axis()
  .scale(y)
  .orient("left")
  .tickSize(-width);


  var zoomBeh = d3.behavior.zoom()
  .x(x)
  .y(y)
  .on("zoom", zoom);

  var svg = d3.select("#scatter")
  .append("svg")
  .attr("width", outerWidth)
  .attr("height", outerHeight)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
  .call(zoomBeh);

  svg.call(tip);

  svg.append("rect")
  .attr("width", width)
  .attr("height", height);

  svg.append("g")
  .classed("x axis", true)
  .attr("transform", "translate(0," + height + ")")
  .call(xAxis)
  .append("text")
  .classed("label", true)
  .attr("x", width)
  .attr("y", margin.bottom - 10)
  .style("text-anchor", "end")
  .text(xCat);

  svg.append("g")
  .classed("y axis", true)
  .call(yAxis)
  .append("text")
  .classed("label", true)
  .attr("transform", "rotate(-90)")
  .attr("y", -margin.left)
  .attr("dy", ".71em")
  .style("text-anchor", "end")
  .text(yCat);

  var objects = svg.append("svg")
  .classed("objects", true)
  .attr("width", width)
  .attr("height", height);

  function zoom() {
    svg.select(".x.axis").call(xAxis);
    svg.select(".y.axis").call(yAxis);

    svg.selectAll(".image")
    .attr("transform", transform);
  }

  var dd = objects.selectAll(".image")
  .data(data).
  enter().append("svg:image")
  .attr('class', 'image')
  // .attr("width", function(d) {return 30;})
  // .attr("height", function(d) {return 30;})
  .attr("width", function(d) {return d.w;})
  .attr("height", function(d) {return d.h;})
  .attr("transform", transform)
  .on("mouseover", tip.show)
  .on("mouseout", tip.hide);

  // console.log(dd)

  return dd
}

function transform(d) {
  return "translate(" + x(d[xCat]) + "," + y(d[yCat]) + ")";
}

var tip = d3.tip()
.attr("class", "d3-tip")
.offset([-10, 0])
.html(function(d) {
  return xCat + ": " + d[xCat] + "<br>" + yCat + ": " + d[yCat];
});


var dataset = 'random100';

var datadir;

if (location.hostname == 'localhost') {
  datadir = '../../../CSHL_cells/gallery/random/'; // <-- Change this to the data folder
} else {
  datadir = '../../../csd395/CSHL_cells/gallery/random/'; // <-- Change this to the data folder
}

var show_blob = true;

load_data(dataset);

function load_data(dataset) {

  // d3.json(datadir+'/'+dataset+'/embedding_'+dataset+'.json'+ '?' + Math.floor(Math.random() * 1000),
  d3.json(datadir+'/'+dataset+'/embedding_'+dataset+'.json',
  function(data) {
    data.forEach(function(d) {
      d.fname_blob = datadir+'/'+dataset+"/blobs/cellMask_"+dataset+"Blobs_"+d.id+"_"+d.index+'.png';
      d.fname_contour = datadir+'/'+dataset+"/contours/cellMask_"+dataset+"Contours_"+d.id+"_"+d.index+'.png';
      // cannot use "width", "height" as keys, may confuse with built-in attributes
    });

    dd = create_data(data)

    dd.attr("xlink:href", function (d) { if (show_blob) {return d.fname_blob;} else {return d.fname_contour;}});

    function toggleContourFunc() {
        show_blob = !show_blob;
        console.log(show_blob);
        dd.attr("xlink:href", function (d) { if (show_blob) {return d.fname_blob;} else {return d.fname_contour;}});
    }

    d3.select('#toggleContour').on('click', toggleContourFunc);
  });
}

d3.select('#changeDataset').on('click', changeDatasetFunc);

function changeDatasetFunc() {
  if (dataset == 'random100'){
    dataset = 'random1000';
    d3.select('svg').remove();
    d3.select('#changeDataset').attr('value', 'Show 100 cells');
    load_data(dataset);
  } else {
    dataset = 'random100';
    d3.select('svg').remove();
    d3.select('#changeDataset').attr('value', 'Show 1000 cells');
    load_data(dataset);
  }
}
