// ---------------------------
// R Wrapper
// ---------------------------
var R = R || {};

// Returns a normally distributed random number (Box–Muller)
R.randn = function(mu, stdev) {
  var u = 0, v = 0;
  while (u === 0) { u = Math.random(); }
  while (v === 0) { v = Math.random(); }
  return mu + stdev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

// Returns a uniformly distributed random number between min and max.
R.randf = function(min, max) {
  return Math.random() * (max - min) + min;
};

// Returns a uniformly distributed integer between min (inclusive) and max (exclusive).
R.randi = function(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
};

// Returns an array of length n filled with zeros.
R.zeros = function(n) {
  return new Array(n).fill(0);
};

// Minimal matrix implementation.
R.Mat = function(rows, cols) {
  this.rows = rows;
  this.cols = cols;
  this.w = new Array(rows * cols).fill(0);
};
R.Mat.prototype.set = function(row, col, value) {
  this.w[row * this.cols + col] = value;
};
R.Mat.prototype.get = function(row, col) {
  return this.w[row * this.cols + col];
};

// R.Graph wrapper. This is used when performing a forward pass and (now)
// when doing backward propagation, we want to call our backend API.
// The constructor accepts a dummy flag (to mimic the original interface).
R.Graph = function(dummyFlag) {
  // The dummyFlag is ignored.
  
  // Sigmoid function: apply elementwise sigmoid to a matrix (assumes property 'w').
  this.sigmoid = function(mat) {
    for (var i = 0; i < mat.w.length; i++) {
      mat.w[i] = 1 / (1 + Math.exp(-mat.w[i]));
    }
    return mat;
  };

  // Backward function: instead of doing computation locally,
  // we call the backend API endpoint '/backward'. This function returns a Promise.
  // Options can be provided (for example, input_values, target, nCycles, learnRate).
  // If not provided, defaults are used (you can modify these as needed).
  this.backward = function(options) {
    // If options is not provided, fill in default values.
    options = options || {};
    // For input_values, if not provided, use a global variable 'data' (or dataBatch if defined)
    if (options.input_values === undefined) {
      if (typeof dataBatch !== 'undefined') {
        options.input_values = dataBatch.w;  // assuming dataBatch is an R.Mat
      } else if (typeof data !== 'undefined' && data.w) {
        options.input_values = data.w;
      } else {
        options.input_values = [];
      }
    }
    // For target, if not provided, use global label data.
    if (options.target === undefined) {
      if (typeof labelBatch !== 'undefined' && labelBatch.w) {
        options.target = labelBatch.w;
      } else if (typeof label !== 'undefined' && label.w) {
        options.target = label.w;
      } else {
        options.target = [];
      }
    }
    // Number of backprop cycles; default is 1.
    if (options.nCycles === undefined) {
      options.nCycles = 1;
    }
    // Learning rate; default is the global learnRate (or 0.01).
    if (options.learnRate === undefined) {
      options.learnRate = (typeof learnRate !== 'undefined') ? learnRate : 0.01;
    }
    
    // Call the backend /backward API.
    return fetch('/backward', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    }).then(function(response) {
      return response.json();
    }).then(function(data) {
      console.log("R.Graph.backward() backend result:", data);
      // Optionally, you could update local state or call a callback here.
      return data;
    }).catch(function(err) {
      console.error("Error in R.Graph.backward():", err);
      throw err;
    });
  };
};

