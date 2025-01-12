/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */

// ---------------------------
// Helper functions (replacing R.*)
// ---------------------------

// Returns a normally distributed random number with mean mu and standard deviation stdev.
function randn(mu, stdev) {
  var u = 0, v = 0;
  while (u === 0) { u = Math.random(); }
  while (v === 0) { v = Math.random(); }
  return mu + stdev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Returns a uniformly distributed random number between min and max.
function randf(min, max) {
  return Math.random() * (max - min) + min;
}

// Returns a uniformly distributed integer between min (inclusive) and max (exclusive).
function randi(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

// Returns an array of length n filled with zero.
function zeros(n) {
  return new Array(n).fill(0);
}

// Minimal matrix implementation.
function Mat(rows, cols) {
  this.rows = rows;
  this.cols = cols;
  // store data in a flat array
  this.w = new Array(rows * cols).fill(0);
}
Mat.prototype.set = function(row, col, value) {
  this.w[row * this.cols + col] = value;
};
Mat.prototype.get = function(row, col) {
  return this.w[row * this.cols + col];
};

// ---------------------------
// Data and NEAT setup variables
// ---------------------------
var md, desktopMode = true;
var dataFactor = 24;
var dataWidth = 320;

function p2d(x) { // pixel to data
  return (x - dataWidth / 2) / dataFactor;
}

function d2p(x) { // data to pixel
  return x * dataFactor + dataWidth / 2;
}

var nBackprop = 600;
var learnRate = 0.01;

var trainer;
var genome, input;
var colaGraph;
var modelReady = false;
var selectedCluster = -1;

// fitness penalties
var penaltyNodeFactor = 0.00;
var penaltyConnectionFactor = 0.03;
var noiseLevel = 0.5;
var makeConnectionProbability = 0.5;
var makeNodeProbability = 0.2;

// presentation mode
var presentationMode = false;

// ---------------------------
// Mobile detection
// ---------------------------
md = new MobileDetect(window.navigator.userAgent);
if (md.mobile()) {
  desktopMode = false;
  console.log('mobile: ' + md.mobile());
  dataWidth = 160;
  dataFactor = 12;
  $("#warningText").show();
} else {
  desktopMode = true;
  console.log('not mobile');
}

// ---------------------------
// Particle and Data Generation Code
// ---------------------------
var Particle = function (x, y, l) {
  this.x = x;
  this.y = y;
  this.l = l;
};

var initNSize = 200;
var nSize = initNSize; // training size
var particleList = [];
var predictionList;
var accuracy = 0.0;
var data;    // training x-data
var label;   // training y-data
var showTrainData = true;

var dataSetChoice = 0;

var initNTestSize = 400;
var nTestSize = initNTestSize; // test size
var particleTestList = [];
var predictionTestList;
var testAccuracy = 0.0;
var testData;    // test x-data
var testLabel;   // test y-data
var showTestData = true;

var nBatch = 10; // minibatch size
var dataBatch = new Mat(nBatch, 2);
var labelBatch = new Mat(nBatch, 1);

var requiredCustomData = 40;

var img;
var imgData;
var imgPrediction;
function createImgData() {
  var x, y;
  var N = (dataWidth / 2 - 1);
  imgData = new Mat(N * N, 2);
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      imgData.set(y * N + x, 0, p2d(2 * x));
      imgData.set(y * N + x, 1, p2d(2 * y));
    }
  }
  imgPrediction = zeros(N * N);
}
createImgData();

var shuffleParticleList = function (pList) {
  var i, randomIndex;
  var N = pList.length;
  for (i = 0; i < N; i++) {
    randomIndex = randi(0, N);
    var temp = pList[i];
    pList[i] = pList[randomIndex];
    pList[randomIndex] = temp;
  }
};

var makeDataLabel = function () {
  var i, n = particleList.length, p;
  predictionList = zeros(n);
  data = new Mat(n, 2);
  label = new Mat(n, 1);
  for (i = 0; i < n; i++) {
    p = particleList[i];
    data.set(i, 0, p.x);
    data.set(i, 1, p.y);
    label.w[i] = p.l;
  }
  n = particleTestList.length;
  predictionTestList = zeros(n);
  testData = new Mat(n, 2);
  testLabel = new Mat(n, 1);
  for (i = 0; i < n; i++) {
    p = particleTestList[i];
    testData.set(i, 0, p.x);
    testData.set(i, 1, p.y);
    testLabel.w[i] = p.l;
  }
};

var makeMiniBatch = function () {
  var i, N = particleList.length, p, randomIndex;
  for (i = 0; i < nBatch; i++) {
    randomIndex = randi(0, N);
    p = particleList[randomIndex];
    dataBatch.set(i, 0, p.x);
    dataBatch.set(i, 1, p.y);
    labelBatch.w[i] = p.l;
  }
};

function generateXORData(numPoints_, noise_) {
  var pList = [];
  var N = (numPoints_ === undefined) ? nSize : numPoints_;
  var noise = (noise_ === undefined) ? 0.5 : noise_;
  for (var i = 0; i < N; i++) {
    var x = randf(-5.0, 5.0) + randn(0, noise);
    var y = randf(-5.0, 5.0) + randn(0, noise);
    var l = 0;
    if (x > 0 && y > 0) l = 1;
    if (x < 0 && y < 0) l = 1;
    pList.push(new Particle(x, y, l));
  }
  return pList;
}

function generateSpiralData(numPoints_, noise_) {
  var pList = [];
  var noise = (noise_ === undefined) ? 0.5 : noise_;
  var N = (numPoints_ === undefined) ? nSize : numPoints_;
  function genSpiral(deltaT, l) {
    var n = N / 2;
    var r, t, x, y;
    for (var i = 0; i < n; i++) {
      r = i / n * 6.0;
      t = 1.75 * i / n * 2 * Math.PI + deltaT;
      x = r * Math.sin(t) + randf(-1, 1) * noise;
      y = r * Math.cos(t) + randf(-1, 1) * noise;
      pList.push(new Particle(x, y, l));
    }
  }
  var flip = randi(0, 2);
  var backside = 1 - flip;
  genSpiral(0, flip);
  genSpiral(Math.PI, backside);
  return pList;
}

function generateGaussianData(numPoints_, noise_) {
  var pList = [];
  var noise = (noise_ === undefined) ? 0.5 : noise_;
  var N = (numPoints_ === undefined) ? nSize : numPoints_;
  function genGaussian(xc, yc, l) {
    var n = N / 2;
    for (var i = 0; i < n; i++) {
      var x = randn(xc, noise * 1.0 + 1.0);
      var y = randn(yc, noise * 1.0 + 1.0);
      pList.push(new Particle(x, y, l));
    }
  }
  genGaussian(2, 2, 1);
  genGaussian(-2, -2, 0);
  return pList;
}

function generateCircleData(numPoints_, noise_) {
  var pList = [];
  var noise = (noise_ === undefined) ? 0.5 : noise_;
  var N = (numPoints_ === undefined) ? nSize : numPoints_;
  var n = N / 2;
  var radius = 5.0;
  function getCircleLabel(x, y) {
    return (x * x + y * y < (radius * 0.5) * (radius * 0.5)) ? 1 : 0;
  }
  for (var i = 0; i < n; i++) {
    var r = randf(0, radius * 0.5);
    var angle = randf(0, 2 * Math.PI);
    var x = r * Math.sin(angle);
    var y = r * Math.cos(angle);
    var noiseX = randf(-radius, radius) * noise / 3;
    var noiseY = randf(-radius, radius) * noise / 3;
    var l = getCircleLabel(x, y);
    pList.push(new Particle(x + noiseX, y + noiseY, l));
  }
  for (i = 0; i < n; i++) {
    r = randf(radius * 0.75, radius);
    angle = randf(0, 2 * Math.PI);
    x = r * Math.sin(angle);
    y = r * Math.cos(angle);
    noiseX = randf(-radius, radius) * noise / 3;
    noiseY = randf(-radius, radius) * noise / 3;
    l = getCircleLabel(x, y);
    pList.push(new Particle(x + noiseX, y + noiseY, l));
  }
  return pList;
}

function generateRandomData() {
  nTestSize = initNTestSize;
  nSize = initNSize;
  var choice = dataSetChoice; 
  if (choice === 0) {
    particleList = generateCircleData(nSize, noiseLevel);
    particleTestList = generateCircleData(nTestSize, noiseLevel);
  } else if (choice === 1) {
    particleList = generateXORData(nSize, noiseLevel);
    particleTestList = generateXORData(nTestSize, noiseLevel);
  } else if (choice === 2) {
    particleList = generateGaussianData(nSize, noiseLevel);
    particleTestList = generateGaussianData(nTestSize, noiseLevel);
  } else {
    particleList = generateSpiralData(nSize, noiseLevel);
    particleTestList = generateSpiralData(nTestSize, noiseLevel);
  }
  makeDataLabel();
}

function alphaColor(c, a) {
  var r = red(c);
  var g = green(c);
  var b = blue(c);
  return color(r, g, b, a);
}

// ---------------------------
// NEAT Related Code
// ---------------------------
function initModel() {
  var i, j;
  N.init({ nInput: 2, nOutput: 1, initConfig: "all",
    activations: "default",
  });
  trainer = new N.NEATTrainer({
    new_node_rate: makeNodeProbability,
    new_connection_rate: makeConnectionProbability,
    sub_population_size: 20,
    init_weight_magnitude: 0.25,
    mutation_rate: 0.9,
    mutation_size: 0.005,
    extinction_rate: 0.5,
  });
  trainer.applyFitnessFunc(fitnessFunc);
  genome = trainer.getBestGenome();
  modelReady = true;
}

function evolveModel(g) {
  // Optionally add random mutation operations.
}

function renderInfo(g) {
  if (typeof g === 'undefined') return;
  var text = "gen: " + N.getNumGeneration() + ", nodes: " + g.getNodesInUse().length + ",\t";
  text += "connections: " + g.connections.length + ",\t";
  if (g.fitness) {
    text += "fitness: " + Math.round(10000 * g.fitness, 0) / 10000 + "<br/>";
  }
  if (presentationMode === false) {
    $("#drawGraph").html(text);
  }
}

async function renderGraph(clusterNum_) {
  var genome;
  if (typeof clusterNum_ !== 'undefined') {
    genome = trainer.getBestGenome(clusterNum_);
  } else {
    genome = trainer.getBestGenome();
  }
  try {
    colaGraph = RenderGraph.getGenomeGraph(genome);
    renderInfo(genome);
    RenderGraph.drawGraph(colaGraph);
  } catch (err) {
    console.error("Error generating genome graph:", err);
  }
  // colaGraph = RenderGraph.getGenomeGraph(genome);
  // renderInfo(genome);
  // RenderGraph.drawGraph(colaGraph);
}

function setCluster(cluster) {
  var K = trainer.num_populations;
  for (var i = 0; i < K; i++) {
    var c = (i === cluster) ? "rgba(0,136,204, 1.0)" : "rgba(0,136,204, 0.15)";
    $("#cluster" + i).css('border-color', c);
  }
  if (typeof cluster !== 'undefined') {
    selectedCluster = cluster;
  }
  renderGraph(cluster);
  calculateAccuracy();
}

// ---------------------------
// p5.js Related Code
// ---------------------------
var myCanvas;
function setup() {
  myCanvas = createCanvas(min($(window).width() * 0.8, 640), min($(window).height() * 0.6, 480));
  myCanvas.parent('p5Container');
  resizeCanvas(dataWidth + 16, dataWidth + 16);
  generateRandomData();
  initModel();
  frameRate(10);
  for (var i = 0; i < 1; i++) {
    trainer.evolve();
    backprop(1);
  }
  renderGraph();
}

function drawDataPoint(p, prediction_) {
  var s = 6;
  var prediction = (typeof prediction_ === 'undefined') ? p.l : prediction_;
  var x = p.x, y = p.y, l = p.l;
  if (prediction === l) {
    if (l === 0) {
      stroke(color(255, 165, 0, 192));
    } else {
      stroke(color(0, 165, 255, 192));
    }
  } else {
    stroke(color(255, 0, 0, 128));
  }
  fill((l === 0) ? alphaColor(color(255, 165, 0), 128) : alphaColor(color(0, 165, 255), 128));
  ellipse(d2p(x), d2p(y), s, s);
}

var fitnessFunc = async function (genome, _backpropMode, _nCycles) {
  "use strict";
  var i;
  var initError, avgError, finalError, veryInitError;
  var nCycles = (_nCycles !== undefined) ? _nCycles : 1;
  var backpropMode = (_backpropMode === true);
  var n = particleList.length;
  
  async function findTotalError() {
    var inputValues = (typeof dataBatch !== 'undefined') ? dataBatch : data;
    try {
      let response = await fetch('/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "input_values": inputValues })
      });
      let result = await response.json();
      var totalError = 0.0;
      var outputs = result.outputs;
      for (i = 0; i < n; i++) {
        var y = outputs[0][i];
        var t = (typeof labelBatch !== 'undefined') ? labelBatch.get(i, 0) : label.get(i, 0);
        var e = -(t * Math.log(y) + (1 - t) * Math.log(1 - y));
        totalError += e;
      }
      return totalError / n;
    } catch (err) {
      console.error("Error in /forward API:", err);
      return 1e20;
    }
  }
  
  initError = await findTotalError();
  avgError = initError;
  veryInitError = initError;
  
  if (!backpropMode) {
    var penaltyNode = genome.getNodesInUse().length - 3;
    var penaltyConnection = genome.connections.length;
    var penaltyFactor = 1 + penaltyNodeFactor * Math.sqrt(penaltyNode) +
                          penaltyConnectionFactor * Math.sqrt(penaltyConnection);
    return -avgError * penaltyFactor;
  }
  
  var genomeBackup = genome.copy();
  var origGenomeBackup = genome.copy();
  var inputValues = (typeof dataBatch !== 'undefined') ? dataBatch : data;
  var targetValues = [];
  var nBatch = (typeof dataBatch !== 'undefined') ? dataBatch.rows : n;
  for (i = 0; i < nBatch; i++) {
    targetValues.push((typeof labelBatch !== 'undefined') ? labelBatch.get(i, 0) : label.get(i, 0));
  }
  
  try {
    let response = await fetch('/backward', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        "input_values": inputValues,
        "target": targetValues,
        "nCycles": nCycles,
        "learnRate": learnRate
      })
    });
    let backResult = await response.json();
    console.log("Backprop final loss =", backResult.loss);
    console.log("Updated weights:", backResult.weights);
  } catch (err) {
    console.error("Error in /backward API:", err);
  }
  
  finalError = await findTotalError();
  if (finalError > initError) {
    avgError = initError;
    genome.copyFrom(genomeBackup);
  } else {
    avgError = finalError;
  }
  if (avgError > veryInitError) {
    avgError = veryInitError;
    genome.copyFrom(origGenomeBackup);
    console.log('backprop was useless.');
  }
  
  var penaltyNode = genome.getNodesInUse().length - 3;
  var penaltyConnection = genome.connections.length;
  var penaltyFactor = 1 + penaltyNodeFactor * Math.sqrt(penaltyNode) +
                          penaltyConnectionFactor * Math.sqrt(penaltyConnection);
  
  return -avgError * penaltyFactor;
};

function buildPredictionList(pList, thedata, thelabel, g, quantisation_) {
  "use strict";
  var i, n, y;
  var output;
  var acc = 0;
  var quantisation = (typeof quantisation_ === "undefined") ? false : quantisation_;
  n = particleList.length;

  // Setup model and forward propagation
  g.setupModel(n);
  g.setInput(thedata);

  return g.forward().then(() => {
    output = g.getOutput();

    for (i = 0; i < n; i++) {
      y = output[i];
      if (!quantisation) {
        pList[i] = (y > 0.5) ? 1.0 : 0.0;
        acc += Math.round(y) === thelabel.get(i, 0) ? 1 : 0;
      } else {
        pList[i] = y;
      }
    }

    acc /= n;
    return acc;
  });
}


function draw() {
  var i, n;
  noStroke();
  fill(255);
  rect(0, 0, width, height);
  textSize(8);
  textFont("Roboto");
  strokeWeight(0.5);
  stroke(64, 192);
  rect(1, 1, width - 1 - 16, height - 1 - 16);
  for (i = -6; i <= 6; i++) {
    strokeWeight(0.5);
    stroke(64, 192);
    if (i > -6 && i < 6) {
      line((i + 6) * (dataWidth / 12) + 1, height - 16, (i + 6) * (dataWidth / 12) + 1, height - 12);
      strokeWeight(0);
      fill(64, 192);
      text("" + (i === 0 ? "X" : i), (i + 6) * (dataWidth / 12) - 1, height - 3);
    }
  }
  for (i = -6; i <= 6; i++) {
    strokeWeight(0.5);
    stroke(64, 192);
    if (i > -6 && i < 6) {
      line(width - 16, (i + 6) * (dataWidth / 12) + 1, width - 12, (i + 6) * (dataWidth / 12) + 1);
      strokeWeight(0);
      fill(64, 192);
      text("" + (i === 0 ? "Y" : -i), width - 10, (i + 6) * (dataWidth / 12) + 3);
    }
  }
  if (modelReady) {
    image(img, 1, 1, dataWidth - 1, dataWidth - 1);
    if (showTrainData) {
      for (i = 0; i < particleList.length; i++) {
        drawDataPoint(particleList[i], predictionList[i]);
      }
    }
    if (showTestData) {
      for (i = 0; i < particleTestList.length; i++) {
        drawDataPoint(particleTestList[i], predictionTestList[i]);
      }
    }
    textSize(10);
    textFont("Courier New");
    stroke(0, 192);
    fill(0, 192);
    if (!presentationMode) {
      if (desktopMode) {
        text("train accuracy = " + Math.round(accuracy * 1000) / 10 + "%\ttest accuracy = " + Math.round(testAccuracy * 1000) / 10 + "%", 8, height - 10 - 14);
      } else {
        text("train accuracy = " + Math.round(accuracy * 1000) / 10 + "%", 8, height - 10 - 14);
      }
    }
  } else {
    // custom data mode
    var remainOrange = requiredCustomData - customDataList[0].length;
    var remainBlue = requiredCustomData - customDataList[1].length;
    textSize(10);
    textFont("Courier New");
    if (customDataMode === 0) {
      stroke(color(255, 165, 0, 128));
      fill(color(255, 165, 0, 128));
    } else {
      stroke(color(0, 165, 255, 128));
      fill(color(0, 165, 255, 128));
    }
    text("Tap in datapoints.\nThe more the better!", 8, 14);
    if (remainOrange > 0) {
      stroke(color(255, 165, 0, 128));
      fill(color(255, 165, 0, 128));
      text("Need " + remainOrange + " more orange datapoint" + (remainOrange > 1 ? "s." : "."), 8, height - 10 - 14 - 14);
    }
    if (remainBlue > 0) {
      stroke(color(0, 165, 255, 128));
      fill(color(0, 165, 255, 128));
      text("Need " + remainBlue + " more blue datapoint" + (remainOrange > 1 ? "s." : "."), 8, height - 10 - 14);
    }
    for (var j = 0; j < 2; j++) {
      for (i = 0; i < customDataList[j].length; i++) {
        drawDataPoint(customDataList[j][i]);
      }
    }
  }
}

// Mouse/touch event handling using jQuery
$("#p5Container").click(function (event) {
  var pos = $("canvas:first").offset();
  var x = event.pageX - pos.left;
  var y = event.pageY - pos.top;
  devicePressed(x, y);
});

$("#spray_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});

$("#clear_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});

function colorClusters() {
  var K = trainer.num_populations;
  var fArray = zeros(K);
  var cArray = zeros(K);
  var best = -1e20;
  var worst = 1e20;
  for (var i = 0; i < K; i++) {
    var f = trainer.getBestGenome(i).fitness;
    best = Math.max(best, f);
    worst = Math.min(worst, f);
    fArray[i] = f;
  }
  var range = Math.max(best - worst, 0.4);
  for (i = 0; i < K; i++) {
    cArray[i] = (fArray[i] - worst) / range;
  }
  for (i = 0; i < K; i++) {
    var level = 0.15 + cArray[i] * 0.85;
    var c = "rgba(0,136,204, " + level + ")";
    $("#cluster" + i).css('background-color', c);
  }
}

function backprop(n, _clusterMode) {
  var clusterMode = (typeof _clusterMode !== 'undefined') ? _clusterMode : true;
  var f = function (g) {
    return (n > 1) ? fitnessFunc(g, true, n) : fitnessFunc(g, false, 1);
  };
  trainer.applyFitnessFunc(f, clusterMode);
  genome = trainer.getBestGenome();
  selectedCluster = (typeof genome.cluster !== 'undefined') ? genome.cluster : -1;
  setCluster(genome.cluster);
  colorClusters();
}

function createPredictionImage() {
  var imgW = dataWidth / 2 - 1;
  img = createImage(imgW, imgW);
  img.loadPixels();
  for (var j = 0; j < img.height; j++) {
    for (var i = 0; i < img.width; i++) {
      var pred = imgPrediction[j * img.height + i];
      var r, g, b;
      if (pred < 0.5) {
        r = 255; g = 165; b = 0;
      } else {
        r = 0; g = 165; b = 255;
      }
      var dist = Math.abs((pred - 0.5) / 0.5);
      img.set(i, j, color(r, g, b, 96 * Math.abs(dist)));
    }
  }
  img.updatePixels();
}

function calculateAccuracy() {
  if (selectedCluster >= 0) {
    genome = trainer.getBestGenome(selectedCluster);
  } else {
    genome = trainer.getBestGenome();
  }
  var bestGenome = trainer.getBestGenome();
  var fitness = fitnessFunc(genome, false, 1);
  var theCluster = genome.cluster;
  genome = trainer.getBestGenome(theCluster);
  accuracy = buildPredictionList(predictionList, data, label, genome);
  testAccuracy = buildPredictionList(predictionTestList, testData, testLabel, genome);
  buildPredictionList(imgPrediction, imgData, null, genome, true);
  createPredictionImage();
}

$("#sgd_button").click(function () {
  $("#controlPanel").fadeOut(500, "swing", function () {
    $("#loadingSpinner").fadeIn(500, "swing", function () {
      backprop(nBackprop);
      $("#loadingSpinner").fadeOut(500, "swing", function () {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });
});

$("#evolve_button").click(function () {
  $("#controlPanel").fadeOut(500, "swing", function () {
    $("#loadingSpinner").fadeIn(500, "swing", function () {
      trainer.evolve();
      backprop(nBackprop);
      $("#loadingSpinner").fadeOut(500, "swing", function () {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });
});

$("#cluster0").click(function () { setCluster(0); });
$("#cluster1").click(function () { setCluster(1); });
$("#cluster2").click(function () { setCluster(2); });
$("#cluster3").click(function () { setCluster(3); });
$("#cluster4").click(function () { setCluster(4); });

$("#warning_button").click(function () { $("#warningText").hide(); });

$(function () {
  $("#sliderNoise").slider({
    max: 0.99,
    min: 0.01,
    step: 0.01,
    value: noiseLevel,
    change: function (event, ui) {
      noiseLevel = ui.value;
      $("#noiseLevel").html("data noise level = " + noiseLevel);
      if (dataSetChoice <= 3) {
        generateRandomData();
      }
    },
  });
});
$(function () {
  $("#sliderNode").slider({
    max: 0.2,
    step: 0.005,
    value: penaltyNodeFactor,
    change: function (event, ui) {
      penaltyNodeFactor = ui.value;
      $("#penaltyNode").html("node count penalty = " + penaltyNodeFactor);
    },
  });
});
$(function () {
  $("#sliderConnection").slider({
    max: 0.2,
    step: 0.005,
    value: penaltyConnectionFactor,
    change: function (event, ui) {
      penaltyConnectionFactor = ui.value;
      $("#penaltyConnection").html("connection count penalty = " + penaltyConnectionFactor);
    },
  });
});
$("#noiseLevel").html("data noise level = " + noiseLevel);
$("#penaltyNode").html("node count penalty = " + penaltyNodeFactor);
$("#penaltyConnection").html("connection count penalty = " + penaltyConnectionFactor);
$(function () {
  $("#sliderBackprop").slider({
    max: 1200,
    min: 100,
    step: 50,
    value: nBackprop,
    change: function (event, ui) {
      nBackprop = ui.value;
      $("#backpropDisplay").html("backprop steps = " + nBackprop);
    },
  });
});
$("#backpropDisplay").html("backprop steps = " + nBackprop);
$(function () {
  $("#sliderLearnRate").slider({
    max: 4,
    min: 0,
    step: 0.01,
    value: 3,
    change: function (event, ui) {
      learnRate = Math.round(Math.pow(10, -(5 - ui.value)) * 100000) / 100000;
      $("#learnRateDisplay").html("learning rate = " + learnRate);
    },
  });
});
$("#learnRateDisplay").html("learning rate = " + learnRate);

// Custom data mode code
var customDataMode = randi(0, 2);
var customDataList = [[], []];
function colorCustomDataChoice() {
  var c0, c1;
  if (customDataMode === 0) {
    c0 = "rgba(255, 165, 0, 0.9)";
    c1 = "rgba(0, 165, 255, 0.4)";
  } else {
    c0 = "rgba(255, 165, 0, 0.4)";
    c1 = "rgba(0, 165, 255, 0.9)";
  }
  $("#customDataOrange").css('border-color', "rgba(255, 165, 0, 1.0)");
  $("#customDataBlue").css('border-color', "rgba(0, 165, 255, 1.0)");
  $("#customDataOrange").css('background-color', c0);
  $("#customDataBlue").css('background-color', c1);
}
function getCustomData() {
  modelReady = false;
  customDataList = [[], []];
  colorCustomDataChoice();
  $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
  $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.4");
  $("#customDataBox").show();
  $("#controlPanel").hide();
}
$("#customDataOrange").click(function () {
  customDataMode = 0;
  colorCustomDataChoice();
});
$("#customDataBlue").click(function () {
  customDataMode = 1;
  colorCustomDataChoice();
});
$("#customDataSubmit").click(function () {
  if (customDataList[0].length >= requiredCustomData && customDataList[1].length >= requiredCustomData) {
    shuffleParticleList(customDataList[0]);
    shuffleParticleList(customDataList[1]);
    var orangeTestIndex = Math.floor(customDataList[0].length / 2);
    var blueTestIndex = Math.floor(customDataList[1].length / 2);
    particleList = customDataList[0].slice(0, orangeTestIndex).concat(customDataList[1].slice(0, blueTestIndex));
    particleTestList = customDataList[0].slice(orangeTestIndex).concat(customDataList[1].slice(blueTestIndex));
    shuffleParticleList(particleList);
    shuffleParticleList(particleTestList);
    nSize = particleList.length;
    nTestSize = particleTestList.length;
    $("#customDataBox").hide();
    $("#controlPanel").show();
    makeDataLabel();
    initModel();
    for (var i = 0; i < 1; i++) {
      trainer.evolve();
      backprop(1);
    }
    renderGraph();
  }
});
function recordTrainingData(x, y) {
  var p = new Particle(x, y, customDataMode);
  customDataList[customDataMode].push(p);
  if (customDataList[0].length >= requiredCustomData && customDataList[1].length >= requiredCustomData) {
    $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
    $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.9");
  }
}
$("#p5Container").click(function (event) {
  var pos = $("canvas:first").offset();
  var x = event.pageX - pos.left;
  var y = event.pageY - pos.top;
  devicePressed(x, y);
});
$("#spray_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});
$("#clear_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});
function setCluster(cluster) {
  var K = trainer.num_populations;
  for (var i = 0; i < K; i++) {
    var c = (i === cluster) ? "rgba(0,136,204, 1.0)" : "rgba(0,136,204, 0.15)";
    $("#cluster" + i).css('border-color', c);
  }
  if (typeof cluster !== 'undefined') {
    selectedCluster = cluster;
  }
  renderGraph(cluster);
  calculateAccuracy();
}
function colorClusters() {
  var K = trainer.num_populations;
  var fArray = zeros(K);
  var cArray = zeros(K);
  var best = -1e20;
  var worst = 1e20;
  for (var i = 0; i < K; i++) {
    var f = trainer.getBestGenome(i).fitness;
    best = Math.max(best, f);
    worst = Math.min(worst, f);
    fArray[i] = f;
  }
  var range = Math.max(best - worst, 0.4);
  for (i = 0; i < K; i++) {
    cArray[i] = (fArray[i] - worst) / range;
  }
  for (i = 0; i < K; i++) {
    var level = 0.15 + cArray[i] * 0.85;
    var c = "rgba(0,136,204, " + level + ")";
    $("#cluster" + i).css('background-color', c);
  }
}
function backprop(n, _clusterMode) {
  var clusterMode = (typeof _clusterMode !== 'undefined') ? _clusterMode : true;
  var f = function (g) {
    return (n > 1) ? fitnessFunc(g, true, n) : fitnessFunc(g, false, 1);
  };
  trainer.applyFitnessFunc(f, clusterMode);
  genome = trainer.getBestGenome();
  selectedCluster = (typeof genome.cluster !== 'undefined') ? genome.cluster : -1;
  setCluster(genome.cluster);
  colorClusters();
}
function createPredictionImage() {
  var imgW = dataWidth / 2 - 1;
  img = createImage(imgW, imgW);
  img.loadPixels();
  for (var j = 0; j < img.height; j++) {
    for (var i = 0; i < img.width; i++) {
      var pred = imgPrediction[j * img.height + i];
      var r, g, b;
      if (pred < 0.5) {
        r = 255; g = 165; b = 0;
      } else {
        r = 0; g = 165; b = 255;
      }
      var dist = Math.abs((pred - 0.5) / 0.5);
      img.set(i, j, color(r, g, b, 96 * Math.abs(dist)));
    }
  }
  img.updatePixels();
}
function calculateAccuracy() {
  if (selectedCluster >= 0) {
    genome = trainer.getBestGenome(selectedCluster);
  } else {
    genome = trainer.getBestGenome();
  }
  var bestGenome = trainer.getBestGenome();
  var fitness = fitnessFunc(genome, false, 1);
  var theCluster = genome.cluster;
  genome = trainer.getBestGenome(theCluster);
  accuracy = buildPredictionList(predictionList, data, label, genome);
  testAccuracy = buildPredictionList(predictionTestList, testData, testLabel, genome);
  buildPredictionList(imgPrediction, imgData, null, genome, true);
  createPredictionImage();
}
$("#sgd_button").click(function () {
  $("#controlPanel").fadeOut(500, "swing", function () {
    $("#loadingSpinner").fadeIn(500, "swing", function () {
      backprop(nBackprop);
      $("#loadingSpinner").fadeOut(500, "swing", function () {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });
});
$("#evolve_button").click(function () {
  $("#controlPanel").fadeOut(500, "swing", function () {
    $("#loadingSpinner").fadeIn(500, "swing", function () {
      trainer.evolve();
      backprop(nBackprop);
      $("#loadingSpinner").fadeOut(500, "swing", function () {
        $("#controlPanel").fadeIn(500, "swing");
      });
    });
  });
});
$("#cluster0").click(function () { setCluster(0); });
$("#cluster1").click(function () { setCluster(1); });
$("#cluster2").click(function () { setCluster(2); });
$("#cluster3").click(function () { setCluster(3); });
$("#cluster4").click(function () { setCluster(4); });
$("#warning_button").click(function () { $("#warningText").hide(); });
$(function () {
  $("#sliderNoise").slider({
    max: 0.99,
    min: 0.01,
    step: 0.01,
    value: noiseLevel,
    change: function (event, ui) {
      noiseLevel = ui.value;
      $("#noiseLevel").html("data noise level = " + noiseLevel);
      if (dataSetChoice <= 3) {
        generateRandomData();
      }
    },
  });
});
$(function () {
  $("#sliderNode").slider({
    max: 0.2,
    step: 0.005,
    value: penaltyNodeFactor,
    change: function (event, ui) {
      penaltyNodeFactor = ui.value;
      $("#penaltyNode").html("node count penalty = " + penaltyNodeFactor);
    },
  });
});
$(function () {
  $("#sliderConnection").slider({
    max: 0.2,
    step: 0.005,
    value: penaltyConnectionFactor,
    change: function (event, ui) {
      penaltyConnectionFactor = ui.value;
      $("#penaltyConnection").html("connection count penalty = " + penaltyConnectionFactor);
    },
  });
});
$("#noiseLevel").html("data noise level = " + noiseLevel);
$("#penaltyNode").html("node count penalty = " + penaltyNodeFactor);
$("#penaltyConnection").html("connection count penalty = " + penaltyConnectionFactor);
$(function () {
  $("#sliderBackprop").slider({
    max: 1200,
    min: 100,
    step: 50,
    value: nBackprop,
    change: function (event, ui) {
      nBackprop = ui.value;
      $("#backpropDisplay").html("backprop steps = " + nBackprop);
    },
  });
});
$("#backpropDisplay").html("backprop steps = " + nBackprop);
$(function () {
  $("#sliderLearnRate").slider({
    max: 4,
    min: 0,
    step: 0.01,
    value: 3,
    change: function (event, ui) {
      learnRate = Math.round(Math.pow(10, -(5 - ui.value)) * 100000) / 100000;
      $("#learnRateDisplay").html("learning rate = " + learnRate);
    },
  });
});
$("#learnRateDisplay").html("learning rate = " + learnRate);

$(function () {
  $("#sliderBackprop").slider({
    max: 1200,
    min: 100,
    step: 50,
    value: nBackprop,
    change: function (event, ui) {
      nBackprop = ui.value;
      $("#backpropDisplay").html("backprop steps = " + nBackprop);
    },
  });
});
$("#backpropDisplay").html("backprop steps = " + nBackprop);

$(function () {
  $("#sliderLearnRate").slider({
    max: 4,
    min: 0,
    step: 0.01,
    value: 3,
    change: function (event, ui) {
      learnRate = Math.round(Math.pow(10, -(5 - ui.value)) * 100000) / 100000;
      $("#learnRateDisplay").html("learning rate = " + learnRate);
    },
  });
});
$("#learnRateDisplay").html("learning rate = " + learnRate);

var customDataMode = randi(0, 2);
var customDataList = [[], []];
function colorCustomDataChoice() {
  var c0, c1;
  if (customDataMode === 0) {
    c0 = "rgba(255, 165, 0, 0.9)";
    c1 = "rgba(0, 165, 255, 0.4)";
  } else {
    c0 = "rgba(255, 165, 0, 0.4)";
    c1 = "rgba(0, 165, 255, 0.9)";
  }
  $("#customDataOrange").css('border-color', "rgba(255, 165, 0, 1.0)");
  $("#customDataBlue").css('border-color', "rgba(0, 165, 255, 1.0)");
  $("#customDataOrange").css('background-color', c0);
  $("#customDataBlue").css('background-color', c1);
}
function getCustomData() {
  modelReady = false;
  customDataList = [[], []];
  colorCustomDataChoice();
  $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
  $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.4");
  $("#customDataBox").show();
  $("#controlPanel").hide();
}
$("#customDataOrange").click(function () {
  customDataMode = 0;
  colorCustomDataChoice();
});
$("#customDataBlue").click(function () {
  customDataMode = 1;
  colorCustomDataChoice();
});
$("#customDataSubmit").click(function () {
  if (customDataList[0].length >= requiredCustomData && customDataList[1].length >= requiredCustomData) {
    shuffleParticleList(customDataList[0]);
    shuffleParticleList(customDataList[1]);
    var orangeTestIndex = Math.floor(customDataList[0].length / 2);
    var blueTestIndex = Math.floor(customDataList[1].length / 2);
    particleList = customDataList[0].slice(0, orangeTestIndex).concat(customDataList[1].slice(0, blueTestIndex));
    particleTestList = customDataList[0].slice(orangeTestIndex).concat(customDataList[1].slice(blueTestIndex));
    shuffleParticleList(particleList);
    shuffleParticleList(particleTestList);
    nSize = particleList.length;
    nTestSize = particleTestList.length;
    $("#customDataBox").hide();
    $("#controlPanel").show();
    makeDataLabel();
    initModel();
    for (var i = 0; i < 1; i++) {
      trainer.evolve();
      backprop(1);
    }
    renderGraph();
  }
});
function recordTrainingData(x, y) {
  var p = new Particle(x, y, customDataMode);
  customDataList[customDataMode].push(p);
  if (customDataList[0].length >= requiredCustomData && customDataList[1].length >= requiredCustomData) {
    $("#customDataSubmit").css('border-color', "rgba(81,163,81, 1.0");
    $("#customDataSubmit").css('background-color', "rgba(81,163,81, 0.9");
  }
}
$("#p5Container").click(function (event) {
  var pos = $("canvas:first").offset();
  var x = event.pageX - pos.left;
  var y = event.pageY - pos.top;
  devicePressed(x, y);
});
$("#spray_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});
$("#clear_button").click(function () {
  generateRandomData();
  initModel();
  renderGraph();
});
$(function () {
  $("#dataChoiceMode").change(function (event) {
    var theChoice = event.target.selectedIndex;
    if (theChoice <= 3) {
      dataSetChoice = theChoice;
      generateRandomData();
      initModel();
      for (var i = 0; i < 1; i++) {
        trainer.evolve();
        backprop(1);
      }
      renderGraph();
    } else if (theChoice === 4) {
      dataSetChoice = theChoice;
      getCustomData();
    } else {
      $("#dataChoiceMode")[0].selectedIndex = dataSetChoice;
      if (dataSetChoice <= 3) {
        generateRandomData();
        calculateAccuracy();
      } else {
        getCustomData();
      }
    }
  });
  $("#dataDisplayMode").change(function (event) {
    var displayMode = event.target.selectedIndex;
    if (displayMode === 0) {
      showTrainData = true;
      showTestData = false;
    } else if (displayMode === 1) {
      showTestData = true;
      showTrainData = false;
    } else {
      showTrainData = true;
      showTestData = true;
    }
  });
});
$("#loadingSpinner").hide();
