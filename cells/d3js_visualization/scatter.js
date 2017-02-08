// reference: http://www.cotrino.com/lifespan/lifespan.js

var margin = { top: 0, right: 0, bottom: 50, left: 50 },
// Ger browser window size http://stackoverflow.com/questions/3333329/javascript-get-browser-height
outerWidth = window.innerWidth - margin.left,
outerHeight = window.innerHeight - margin.bottom,
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
  return "member count = " + d["memberCount"];
  // return xCat + ": " + d[xCat] + "<br>" + yCat + ": " + d[yCat];
});



// var dataset = 'random10';
var dataset = 'largest30p'
var cell_number = '100'
var nystrom_sample_num = '100'

var datadir;

datadir = '../../../CSHL_cells_v2/d3js/';

// if (location.hostname == 'localhost') {
//   datadir = '/home/yuncong/CSHL_cells_v2/d3js/'; // <-- Change this to the data folder
// } else {
//   datadir = '/home/yuncong/csd395/CSHL_cells_v2/d3js/'; // <-- Change this to the data folder
// }

var show_blob = true;

load_data(dataset, cell_number, nystrom_sample_num);

function load_data(dataset, cell_number=10, nystrom_sample_num=200) {

  // d3.json(datadir+'/'+dataset+'/embedding_'+dataset+'.json'+ '?' + Math.floor(Math.random() * 1000),
  d3.json(datadir+dataset+'/random'+cell_number+'/embedding_random'+cell_number+'_nystrom'+nystrom_sample_num+'.json',
  function(data) {
    data.forEach(function(d) {
      d.fname_blob = datadir+'/'+dataset+'/random'+cell_number+"/blobs/cellMask_random"+cell_number+"Blobs_"+d.id+"_"+d.index+'.png';
      d.fname_contour = datadir+'/'+dataset+'/random'+cell_number+"/contours/cellMask_random"+cell_number+"Contours_"+d.id+"_"+d.index+'.png';
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
//
// d3.select('#changeDataset').on('click', changeDatasetFunc);
//
// function changeDatasetFunc() {
//   if (dataset == 'random100'){
//     dataset = 'random1000';
//     d3.select('svg').remove();
//     d3.select('#changeDataset').attr('value', 'Show 100 cells');
//     load_data(dataset);
//   } else {
//     dataset = 'random100';
//     d3.select('svg').remove();
//     d3.select('#changeDataset').attr('value', 'Show 1000 cells');
//     load_data(dataset);
//   }
// }


// Get selection from combox box http://stackoverflow.com/questions/18883675/d3-js-get-value-of-selected-option

d3.select('#dataset_list').on('change', dataset_changed);

function dataset_changed() {

  value = d3.select('#dataset_list').property("value");
  dataset = value;
  d3.select('svg').remove();
  load_data(dataset, cell_number, nystrom_sample_num);
}


d3.select('#cell_number_list').on('change', cell_number_changed);

function cell_number_changed() {

  value = d3.select('#cell_number_list').property("value");
  cell_number = value;
  d3.select('svg').remove();
  load_data(dataset, cell_number, nystrom_sample_num);
}

d3.select('#nystrom_sample_number_list').on('change', nystrom_sample_number_changed);

function nystrom_sample_number_changed() {
  value = d3.select('#nystrom_sample_number_list').property("value");
  nystrom_sample_num = value;
  d3.select('svg').remove();
  load_data(dataset, cell_number, nystrom_sample_num);
}


d3.select('#x_axis_label').on('change', x_label_changed);

function x_label_changed() {
  value = d3.select('#x_axis_label').property("value");
  xCat = value;
  d3.select('svg').remove();
  load_data(dataset, cell_number, nystrom_sample_num);
}

d3.select('#y_axis_label').on('change', y_label_changed);

function y_label_changed() {
  value = d3.select('#y_axis_label').property("value");
  yCat = value;
  d3.select('svg').remove();
  load_data(dataset, cell_number, nystrom_sample_num);
}
