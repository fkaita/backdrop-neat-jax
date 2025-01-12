#!/usr/bin/env python
"""
This file sets up the Flask server for the NEAT implementation.
It imports all necessary classes and functions from neat.py and defines API endpoints.
"""

from flask import Flask, request, jsonify, send_from_directory
import math
import json
import jax.numpy as jnp

# Import everything from neat.py
from neat import (
    init, NEATTrainer, Genome,
    nInput, nOutput, initConfig,
    connections, nodes, generationNum
)

app = Flask(__name__, static_folder='static')

# Global instances (for simplicity, we allow one instance of each)
global_trainer = None

@app.route('/')
def index():
    """ Serve the index.html file from static/ """
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/init", methods=["POST"])
def init_network():
    """
    Initialize the NEAT network using options provided as JSON.
    Example JSON:
    {
      "nInput": 2,
      "nOutput": 1,
      "initConfig": "one",
      "activations": "minimal"
    }
    """
    opts = request.get_json() or {}
    init(opts)
    return jsonify({"status": "initialized", "nInput": nInput, "nOutput": nOutput, "initConfig": initConfig})

@app.route("/network", methods=["GET"])
def network_api():
    """
    Returns current global network parameters.
    Output JSON structure:
      {
         "nInput": <nInput value>,
         "nOutput": <nOutput value>,
         "nodes": <list of nodes>,
         "connections": <list of connections>
      }
    """
    return jsonify({
        "nInput": nInput,
        "nOutput": nOutput,
        "nodes": nodes,
        "connections": connections
    })


@app.route("/create_genome", methods=["GET"])
def create_genome_api():
    """
    Create a new Genome.
    """
    genome = Genome()
    genome.addRandomConnection()
    genome.mutateWeights(1.0, 1.0)
    return jsonify({"genome": genome.toJSON("new genome")})

@app.route("/create_trainer", methods=["POST"])
def create_trainer_api():
    """
    Create a new NEATTrainer. Options may be provided in JSON.
    """
    global global_trainer
    opts = request.get_json() or {}
    global_trainer = NEATTrainer(options=opts)
    return jsonify({"status": "trainer created", "num_genes": len(global_trainer.genes)})

@app.route("/evolve", methods=["POST"])
def evolve_api():
    """
    Perform one evolution step.
    Optionally, you can provide {"mutateWeightsOnly": true} in the JSON.
    """
    global global_trainer
    if global_trainer is None:
        return jsonify({"error": "trainer not created"}), 400
    data = request.get_json() or {}
    mutateOnly = data.get("mutateWeightsOnly", False)
    global_trainer.evolve(mutateOnly)
    return jsonify({"status": "evolution step completed", "generation": generationNum, "num_genes": len(global_trainer.genes)})

@app.route("/best_genome", methods=["GET"])
def best_genome_api():
    """
    Return the best genome (as JSON) from the trainer.
    """
    global global_trainer
    if global_trainer is None:
        return jsonify({"error": "trainer not created"}), 400
    best = global_trainer.getBestGenome()
    return jsonify({"best_genome": best.toJSON("best genome")})

@app.route("/forward", methods=["POST"])
def forward_api():
    """
    Perform a forward pass on a given genome using provided input values.
    
    Expected JSON input:
    {
      "genome": <genome JSON>,          // optional; if provided, use this genome; otherwise, use the best genome.
      "input_values": [list of input values]   // optional; if not provided, defaults to 0.5 for each input node.
    }
    
    Returns JSON:
      { "outputs": [list of outputs] }
    """
    data = request.get_json() or {}
    input_values = data.get("input_values", None)
    if input_values is not None:
        input_values = jnp.array(input_values)
    if "genome" in data:
        genome_json = data["genome"]
        genome_obj = Genome()
        genome_obj.fromJSON(json.dumps(genome_json))
    else:
        global global_trainer
        if global_trainer is None:
            return jsonify({"error": "trainer not created"}), 400
        genome_obj = global_trainer.getBestGenome()
    outputs = genome_obj.forward(input_values=input_values)
    return jsonify({"outputs": outputs.tolist()})

@app.route("/backward", methods=["POST"])
def backward_api():
    """
    Compute gradient descent (backpropagation) on the best genome's connection weights.
    
    Expected JSON input:
    {
      "input_values": [list of input values],   // optional; defaults to 0.5 for each input
      "target": [list of target output values],   // required; used to compute the squared error loss
      "nCycles": <number of backprop cycles>,     // optional; defaults to 1
      "learnRate": <learning rate>                  // optional; defaults to 0.01
    }
    
    For each cycle:
      loss = sum((output - target)^2)
      new_weight = current_weight - learnRate * gradient
    
    Returns JSON:
      {
         "loss": <final scalar loss>,
         "weights": [updated connection weights as a list]
      }
    """
    global global_trainer
    if global_trainer is None:
        return jsonify({"error": "trainer not created"}), 400

    data = request.get_json() or {}
    target = data.get("target", None)
    if target is None:
        return jsonify({"error": "target must be provided"}), 400

    input_values = data.get("input_values", None)
    if input_values is not None:
        input_values = jnp.array(input_values)

    nCycles = data.get("nCycles", 1)
    learnRate = data.get("learnRate", 0.01)

    def loss_fn(outputs):
        t = jnp.array(target)
        return jnp.sum((outputs - t) ** 2)

    genome = global_trainer.getBestGenome()
    for cycle in range(nCycles):
        loss_val, grad_weights = genome.backward(loss_fn, input_values=input_values)
        current_weights = jnp.array([c[IDX_WEIGHT] for c in genome.connections])
        new_weights = current_weights - learnRate * grad_weights
        new_weights_list = new_weights.tolist()
        for index, w in enumerate(new_weights_list):
            genome.connections[index][IDX_WEIGHT] = w

    return jsonify({
        "loss": float(loss_val),
        "weights": [c[IDX_WEIGHT] for c in genome.connections]
    })

if __name__ == '__main__':
    # Optionally initialize the network with default settings.
    init({"nInput": 2, "nOutput": 1, "initConfig": "one", "activations": "minimal"})
    app.run(host="0.0.0.0", port=5000, debug=True)
