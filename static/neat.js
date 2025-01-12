// Define or extend the global N object.
var N = N || {};

// -----------------------------
// Network Management
// -----------------------------

// Get all network parameters (nInput, nOutput, nodes, connections)
N.getNetworkParameters = function() {
  return fetch("/network", {
    method: "GET",
    headers: { "Content-Type": "application/json" }
  }).then(response => response.json());
};

// Convenience method: get number of input nodes.
N.getNumInput = function() {
  return N.getNetworkParameters().then(data => data.nInput);
};

// Convenience method: get number of output nodes.
N.getNumOutput = function() {
  return N.getNetworkParameters().then(data => data.nOutput);
};

// Convenience method: get nodes list.
N.getNodes = function() {
  return N.getNetworkParameters().then(data => data.nodes);
};

// Convenience method: get connections list.
N.getConnections = function() {
  return N.getNetworkParameters().then(data => data.connections);
};

// Initialize the network (calls backend /init)
N.init = function(options) {
  return fetch("/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options)
  }).then(function(response) {
    return response.json();
  });
};

// -----------------------------
// TrainerWrapper Class
// -----------------------------
function TrainerWrapper(options) {
  // Create the trainer on the backend and store any metadata if needed.
  this.ready = fetch("/create_trainer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options)
  })
    .then(response => response.json())
    .then(data => {
      this.meta = data;
      return data;
    });
}

TrainerWrapper.prototype.applyFitnessFunc = function(fitnessFunc, clusterMode = true) {
  const fitnessConfig = fitnessFunc.toString(); // Serialize the fitness function.
  return fetch("/apply_fitness", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      fitness_config: fitnessConfig,
      cluster_mode: clusterMode 
    })
  }).then(response => response.json());
};

TrainerWrapper.prototype.getBestGenome = function(cluster) {
  let url = "/best_genome";
  if (typeof cluster !== "undefined") {
    url += `?cluster=${cluster}`;
  }
  return fetch(url, { method: "GET" })
    .then(response => response.json())
    .then(data => new GenomeWrapper(data.best_genome));
};

TrainerWrapper.prototype.evolve = function(mutateWeightsOnly = false) {
  return fetch("/evolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mutateWeightsOnly: mutateWeightsOnly })
  }).then(response => response.json());
};

// -----------------------------
// GenomeWrapper Class
// -----------------------------
function GenomeWrapper(genomeData) {
  this.data = genomeData;
}

GenomeWrapper.prototype.getNodesInUse = function() {
  return this.data.nodes || [];
};

Object.defineProperty(GenomeWrapper.prototype, "connections", {
  get: function() {
    return this.data.connections || [];
  }
});

GenomeWrapper.prototype.copy = function() {
  return new GenomeWrapper(JSON.parse(JSON.stringify(this.data)));
};

GenomeWrapper.prototype.copyFrom = function(otherGenomeWrapper) {
  this.data = JSON.parse(JSON.stringify(otherGenomeWrapper.data));
};

// -----------------------------
// Integration with Existing N API
// -----------------------------

N.NEATTrainer = TrainerWrapper;
N.Genome = GenomeWrapper;

// Example usage (optional):
// const trainer = new N.NEATTrainer({
//   new_node_rate: 0.3,
//   new_connection_rate: 0.5,
//   sub_population_size: 20,
//   init_weight_magnitude: 0.25,
//   mutation_rate: 0.9,
//   mutation_size: 0.01,
//   extinction_rate: 0.5
// });
// trainer.ready.then(() => {
//   return trainer.applyFitnessFunc(myFitnessFunction);
// });
