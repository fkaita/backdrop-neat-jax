#!/usr/bin/env python3
# server.py
import os
from flask import Flask, request, jsonify, send_from_directory
from jax import random
import jax.numpy as jnp

# Import your neat.py
from neat import (
    Genome,
    init_genome,
    forward,
    mutate_genome,
    crossover,
    # ... any other functions you need ...
)

app = Flask(__name__, static_folder='static')

# We'll store a "population" of NEAT Genomes globally:
global_population = []
rng = random.PRNGKey(42)

@app.route('/')
def index():
    """ Serve the index.html file from static/ """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/init', methods=['POST'])
def init_pop():
    """
    Initialize a population of NEAT genomes.
    Expects JSON like: { "n_input": 2, "n_output": 1, "init_config": "all", "pop_size": 10 }
    """
    global global_population, rng
    req = request.get_json()
    n_input     = req.get("n_input", 2)
    n_output    = req.get("n_output", 1)
    init_config = req.get("init_config", "all")
    pop_size    = req.get("pop_size", 10)

    new_pop = []
    for i in range(pop_size):
        rng, key = random.split(rng)
        g = init_genome(key, n_input, n_output, init_config)
        new_pop.append(g)

    global_population = new_pop
    return jsonify({"msg": f"Initialized population of size {pop_size}."})

@app.route('/forward', methods=['POST'])
def forward_pass():
    """
    Forward pass on a single genome from the population.
    Expects JSON:
      { "genome_index": 0,
        "inputs": [[x1, y1], [x2, y2], ...]  // batch
      }
    Returns predicted outputs as a 2D list (batch_size x n_outputs).
    """
    global global_population
    req = request.get_json()
    idx = req.get("genome_index", 0)
    inputs = jnp.array(req.get("inputs", []), dtype=jnp.float32)  # shape [batch, 2] typically

    if idx < 0 or idx >= len(global_population):
        return jsonify({"error": "Invalid genome index"}), 400

    outputs = forward(global_population[idx], inputs)
    return jsonify({"outputs": outputs.tolist()})

# ... your existing endpoints (forward, mutate, crossover, etc.) ...

@app.route('/backprop', methods=['POST'])
def backprop_endpoint():
    """
    JSON: {
      "genome_index": 0,
      "inputs": [[0,0],[1,0],...],
      "labels": [0, 1, ...],
      "n_steps": 100,
      "lr": 0.05
    }
    Returns updated genome index or info.
    """
    global global_population, rng

    data = request.get_json()
    idx = data.get("genome_index", 0)
    x   = jnp.array(data["inputs"], dtype=jnp.float32)
    y   = jnp.array(data["labels"], dtype=jnp.float32)
    steps = data.get("n_steps", 100)
    lr    = data.get("lr", 0.05)

    if idx < 0 or idx >= len(global_population):
        return jsonify({"error": "Invalid index"}), 400

    g = global_population[idx]

    rng, subkey = random.split(rng)
    # run backprop
    g_new = backprop_genome(subkey, g, x, y, n_steps=steps, lr=lr)

    # store updated genome
    global_population[idx] = g_new

    # optionally, return some info
    out_before = forward(g, x)
    out_after  = forward(g_new, x)
    return jsonify({
        "msg": f"Backprop complete on genome {idx}",
        "before": out_before.tolist(),
        "after":  out_after.tolist()
    })

@app.route('/mutate', methods=['POST'])
def mutate():
    """
    Mutate a genome in the population.
    Expects JSON:
      { "genome_index": 0,
        "mutation_rate": 0.2,
        "mutation_scale": 0.5
      }
    """
    global global_population, rng
    req = request.get_json()
    idx = req.get("genome_index", 0)
    mrate = req.get("mutation_rate", 0.2)
    mscale= req.get("mutation_scale", 0.5)

    if idx < 0 or idx >= len(global_population):
        return jsonify({"error": "Invalid genome index"}), 400

    rng, key = random.split(rng)
    mutated = mutate_genome(key, global_population[idx], mrate, mscale)
    global_population[idx] = mutated
    return jsonify({"msg": f"Mutated genome {idx}."})

@app.route('/crossover', methods=['POST'])
def do_crossover():
    """
    Crossover between two genomes in population to produce a new child genome.
    Expects JSON:
      { "mom_index": 0, "dad_index": 1 }
    Appends the child to the population, returns child's index.
    """
    global global_population, rng
    req = request.get_json()
    mom_idx = req.get("mom_index", 0)
    dad_idx = req.get("dad_index", 0)

    if not (0 <= mom_idx < len(global_population)) or not (0 <= dad_idx < len(global_population)):
        return jsonify({"error": "Invalid mom/dad index"}), 400

    rng, key = random.split(rng)
    child = crossover(key, global_population[mom_idx], global_population[dad_idx])
    global_population.append(child)
    child_idx = len(global_population) - 1

    return jsonify({
        "msg": f"Created child at index {child_idx}",
        "child_index": child_idx
    })

# Example "evaluate" endpoint (if you had a fitness function):
@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Very simple placeholder if you want to compute fitness for each genome.
    Expects JSON with some data, e.g. { "input_data": [[x,y],...], "labels": [...], ... }
    Then it returns a list of fitnesses for all genomes.
    """
    global global_population
    req = request.get_json()
    input_data = jnp.array(req.get("input_data", []), dtype=jnp.float32)
    labels     = jnp.array(req.get("labels", []), dtype=jnp.float32)
    # You'd do a real loop or vectorized approach:
    fitnesses = []
    for g in global_population:
        preds = forward(g, input_data)  # shape [batch, 1] for example
        # compute error vs. labels => fitness
        error = jnp.mean((preds[:,0] - labels)**2)
        fitness = float(-error)  # NEAT typically wants higher = better
        fitnesses.append(fitness)
    # You might store them or do something else
    return jsonify({"fitnesses": fitnesses})

@app.route('/graph', methods=['POST'])
def get_graph():
    """
    Return a JSON for the genome's graph, usable by RenderGraph.drawGraph().
    Expects JSON:
      {
        "genome_index": 0,
        "n_input": 2,
        "n_output": 1
      }
    """
    global global_population
    req = request.get_json()
    idx = req.get("genome_index", 0)
    n_input = req.get("n_input", 2)
    n_output= req.get("n_output", 1)

    if idx < 0 or idx >= len(global_population):
        return jsonify({"error": "Invalid genome index"}), 400

    g = global_population[idx]
    graph_json = genome_to_rendergraph(g, n_input, n_output)

    return jsonify({"graph": graph_json})


if __name__ == "__main__":
    # By default, runs on localhost:5000
    app.run(port=5000, debug=False)
