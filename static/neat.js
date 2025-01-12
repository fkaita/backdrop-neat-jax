// Define a global N object that “simulates” the original NEAT interface by calling backend API endpoints.
var N = {};

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

// Create a trainer (calls backend /create_trainer)
// In the original code, "new N.NEATTrainer(...)" would create a trainer locally.
// Here, we wrap the API call and simply return the backend’s response.
// (Since your backend uses a global trainer instance, you don’t need to return an object with methods.)
N.NEATTrainer = function(options) {
  return fetch("/create_trainer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options)
  }).then(function(response) {
    return response.json();
  });
};

// Evolve the trainer (calls backend /evolve)
// This function accepts a boolean flag for mutateWeightsOnly.
N.evolve = function(mutateWeightsOnly) {
  return fetch("/evolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mutateWeightsOnly: mutateWeightsOnly })
  }).then(function(response) {
    return response.json();
  });
};

// Get the best genome (calls backend /best_genome)
// Returns a promise that resolves to the best genome object.
N.bestGenome = function() {
  return fetch("/best_genome", { method: "GET" })
    .then(function(response) {
      return response.json();
    })
    .then(function(data) {
      // Assume the backend returns an object like { "best_genome": { ... } }
      return data.best_genome;
    });
};

// (Optional) For example, if you need the generation number:
N.getNumGeneration = function() {
  // You might include a generation field in your genome JSON.
  return N.bestGenome().then(function(genome) {
    return genome.generation || 0;
  });
};
