#!/usr/bin/env python3

import jax
print(jax.devices())  
import jax.numpy as jnp
from jax import random, jit

######################################################
# NODE TYPES & ACTIVATIONS
######################################################
# For simplicity, define a small set:
NODE_INPUT    = 0
NODE_OUTPUT   = 1
NODE_BIAS     = 2
NODE_RELU     = 3
NODE_TANH     = 4

# Map node-type -> activation function
def activate(node_type, x):
    """
    node_type is a JAX integer tracer. We use lax.switch to do 
    a 'switch'-like operation: node_type picks which function to call.
    We'll define the function index as:
      0 -> (NODE_INPUT)
      1 -> (NODE_OUTPUT)
      2 -> (NODE_BIAS)
      3 -> (NODE_RELU)
      4 -> (NODE_TANH)
    But you must make sure node_type is in [0..4].
    """
    def f_input(x):  return x  # pass-through
    def f_output(x): return x
    def f_bias(x):   return x
    def f_relu(x):   return jnp.maximum(0, x)
    def f_tanh(x):   return jnp.tanh(x)

    # define a tuple of possible “cases”:
    branches = (f_input, f_output, f_bias, f_relu, f_tanh)

    # use jax.lax.switch(index, branches, operand)
    return jax.lax.switch(node_type, branches, x)
######################################################
# GENOME DEFINITION
######################################################
# We store everything inside a Python dataclass or dictionary.
# Let’s use a namedtuple or dataclass for clarity.

from typing import NamedTuple

class Genome(NamedTuple):
    """
    node_types: jnp.int32 array [n_nodes]. E.g. [0,0,2,1,1,...]
      (some inputs, 1 bias, then some outputs, maybe hidden nodes).
    connections: jnp.int32 array [n_connections, 2].
      Each row: [from_node, to_node].
    weights: jnp.float32 array [n_connections].
      Each entry is the weight for that connection.
    active: jnp.int32 array [n_connections].
      1 if active, 0 if disabled.
    """
    node_types:   jnp.ndarray
    connections:  jnp.ndarray
    weights:      jnp.ndarray
    active:       jnp.ndarray

######################################################
# INITIALIZATION
######################################################
def init_genome(rng, n_input, n_output, init_config="none"):
    """
    Create a simple genome with:
      - node_types: input nodes, 1 bias node, output nodes (no hidden nodes).
      - connections: possibly none, or fully connected, etc.
    """
    # 1) Build node_types
    # Example: first n_input => input nodes, then 1 => bias, then n_output => output
    node_types = [NODE_INPUT]*n_input + [NODE_BIAS] + [NODE_OUTPUT]*n_output
    node_types = jnp.array(node_types, dtype=jnp.int32)

    # 2) Build connections, weights, active
    connections = []
    weights     = []
    active      = []

    if init_config == "all":
        # fully connect input+bias to all outputs
        for i in range(n_input + 1):
            for j in range(n_output):
                from_idx = i
                to_idx   = n_input + 1 + j  # because outputs start after the bias
                connections.append([from_idx, to_idx])
                # random normal weight
                w = random.normal(rng, (1,)) * 1.0
                weights.append(w[0])
                active.append(1)
    elif init_config == "one":
        # connect only one hidden node, for illustration
        # but let's keep it minimal: just connect bias -> first output
        bias_idx   = n_input
        out0_idx   = n_input + 1  # first output
        connections.append([bias_idx, out0_idx])
        w = random.normal(rng, (1,)) * 1.0
        weights.append(w[0])
        active.append(1)
    # else "none" => empty

    if len(connections) == 0:
        connections_array = jnp.zeros((0,2), dtype=jnp.int32)
        weights_array     = jnp.zeros((0,),  dtype=jnp.float32)
        active_array      = jnp.zeros((0,),  dtype=jnp.int32)
    else:
        connections_array = jnp.array(connections, dtype=jnp.int32)
        weights_array     = jnp.array(weights,     dtype=jnp.float32)
        active_array      = jnp.array(active,      dtype=jnp.int32)

    return Genome(
        node_types=node_types,
        connections=connections_array,
        weights=weights_array,
        active=active_array
    )

######################################################
# FORWARD PASS
######################################################
def forward(genome: Genome, inputs: jnp.ndarray):
    """
    - Suppose inputs is shape [n_input] or [batch_size, n_input].
    - We do a naive "tick" approach:
        1) node_outputs[i] = 0 for all i initially,
        2) place the 'inputs' in the input nodes,
        3) place bias=1.0 in the bias node,
        4) repeatedly propagate signals in topological order,
           or just do a fixed iteration if we assume no cycles.
      For a *real* NEAT approach, you'd do a topological sort
      or BFS to handle hidden nodes in order.
    Here, we keep it extremely simplified and assume no hidden
    cycles. We do enough ticks to let signals flow from input
    to output.
    """
    n_nodes = genome.node_types.shape[0]

    # If inputs is 1D, reshape it to [1, n_input] so batch logic is simpler
    if inputs.ndim == 1:
        inputs = inputs[None, :]  # [1, n_input]
    batch_size = inputs.shape[0]

    # We'll store node activations in a [batch_size, n_nodes] array
    node_vals = jnp.zeros((batch_size, n_nodes), dtype=jnp.float32)

    n_input  = jnp.sum(genome.node_types == NODE_INPUT)
    bias_idx = n_input            # if we assume the bias is right after the inputs
    # place input and bias
    def place_io(node_vals):
        # slice assignment in JAX requires fancy indexing or use of scatter
        # simpler approach: build an array with them set
        node_vals = node_vals.at[:, :n_input].set(inputs)  # place inputs
        node_vals = node_vals.at[:, bias_idx].set(1.0)     # place bias
        return node_vals

    node_vals = place_io(node_vals)

    # We can do multiple “ticks” to allow signals to propagate
    # (in real NEAT code, we do a topological pass)
    def one_tick(node_vals):
        # For each connection, do:
        #   node_vals[to] += node_vals[from] * weight
        # Then apply activation for each node
        from_nodes = genome.connections[:, 0]  # shape [n_connections]
        to_nodes   = genome.connections[:, 1]  # shape [n_connections]
        w          = genome.weights            # shape [n_connections]
        actv       = genome.active            # shape [n_connections]

        # We'll accumulate contributions in a buffer, then add to node_vals
        contrib_buffer = jnp.zeros_like(node_vals)  # [batch, n_nodes]

        # Vectorized approach:
        #   For each connection c, let from_nodes[c] be f,
        #   to_nodes[c] be t, and weight w[c].
        #   Then add node_vals[:, f] * w[c] to contrib_buffer[:, t].
        # Because we can't do arbitrary scatter easily in pure jnp
        # with direct indexing, we can use `jax.ops.segment_sum` or
        # `scatter_add`. We'll do a manual approach:

        def body_fun(i, buf):
            # connection i
            f = from_nodes[i]
            t = to_nodes[i]
            aw= actv[i] * w[i]
            # add contribution to buf[:, t]
            contribution = node_vals[:, f] * aw
            return buf.at[:, t].add(contribution)

        contrib_buffer = jax.lax.fori_loop(0, genome.connections.shape[0], body_fun, contrib_buffer)

        # Add the contributions
        new_vals = node_vals + contrib_buffer

        # Now apply activation per node
        # We can do it in a vectorized way as well:
        def apply_node_activation(i, new_vals):
            nt = genome.node_types[i]
            return new_vals.at[:, i].set(activate(nt, new_vals[:, i]))

        new_vals = jax.lax.fori_loop(0, n_nodes, apply_node_activation, new_vals)
        return new_vals

    # We do a few ticks:
    for _ in range(3):
        node_vals = one_tick(node_vals)

    # Finally extract the output(s).
    # Let's say all nodes with node_type == NODE_OUTPUT are outputs:
    out_mask = (genome.node_types == NODE_OUTPUT)
    # we can gather them:
    output_nodes = jnp.where(out_mask)[0]  # indices of outputs
    # shape = [batch_size, #outputs]
    output_vals = node_vals[:, output_nodes]
    return output_vals

######################################################
# BACKPROP FUNCTION
######################################################
def backprop_genome(rng,
                    genome: Genome,
                    inputs: jnp.ndarray,
                    labels: jnp.ndarray,
                    n_steps: int=1,
                    lr: float=0.01):
    """
    Perform 'n_steps' of gradient descent on 'genome.weights'
    to minimize MSE between forward(genome, inputs) and labels.
    We'll assume shape of inputs=[batch, n_input], labels=[batch, n_output].
    """
    # 1) define a loss function that depends on 'weights'
    #    We'll copy 'genome' but swap in the candidate weights.
    def genome_loss(weights, inputs, labels, genome):
        # build a "temporary" genome with these weights
        tmp_genome = genome._replace(weights=weights)
        preds = forward(tmp_genome, inputs)  # shape [batch, n_output]
        # for MSE:
        # ensure labels matches shape [batch, n_output]
        if labels.ndim == 1:
            # expand to [batch,1]
            labels_2d = labels[:, None]
        else:
            labels_2d = labels
        mse = jnp.mean((preds - labels_2d)**2)
        return mse

    # 2) get gradient w.r.t. 'weights'
    grad_loss = jax.grad(genome_loss, argnums=0)

    # 3) do n_steps of gradient descent
    new_weights = genome.weights
    for _ in range(n_steps):
        g = grad_loss(new_weights, inputs, labels, genome)
        new_weights = new_weights - lr * g

    # 4) return updated genome
    return genome._replace(weights=new_weights)

######################################################
# MUTATION
######################################################
def mutate_genome(rng, genome: Genome, mutation_rate=0.2, mutation_scale=0.5):
    """
    - With probability mutation_rate, perturb each weight
      by a normal(0, mutation_scale).
    - Potentially also add new connections or new nodes, etc.
      (We keep it minimal for illustration.)
    """
    # 1) mutate weights
    key1, rng = random.split(rng)
    do_perturb = random.uniform(key1, shape=genome.weights.shape) < mutation_rate

    key2, rng = random.split(rng)
    deltas = random.normal(key2, shape=genome.weights.shape) * mutation_scale

    new_weights = jnp.where(do_perturb, genome.weights + deltas, genome.weights)

    # 2) optionally add more advanced mutations (new node, new connection) ...
    #   for brevity, we skip that.

    return genome._replace(weights=new_weights)

######################################################
# CROSSOVER
######################################################
def crossover(rng, mom: Genome, dad: Genome):
    """
    Simple 1-to-1 “aligned” crossover for demonstration:
    - We assume mom.connections == dad.connections (same shape).
    - For each weight, 50% chance to come from mom or dad.
    If their topologies differ, you'd need more complex logic.
    """
    # For simplicity, we just check shapes are the same:
    if mom.connections.shape != dad.connections.shape:
        # real NEAT code does fancy matching by global innovation IDs,
        # we skip that here. We'll just return a copy of mom if mismatched
        return mom

    # choose from mom or dad
    key1, rng = random.split(rng)
    mask = random.bernoulli(key1, 0.5, shape=mom.weights.shape)
    new_weights = jnp.where(mask, mom.weights, dad.weights)

    # If they differ in “active”, do a similar approach or logical OR, etc.
    # Here, we do a simple approach: if both active => active,
    # else if one is active => 50% chance:
    both_active = jnp.logical_and(mom.active == 1, dad.active == 1)
    either_active = jnp.logical_or(mom.active == 1, dad.active == 1)
    key2, rng = random.split(rng)
    coin_flip = random.bernoulli(key2, 0.5, shape=mom.active.shape)
    new_active = jnp.where(
        both_active,
        1,
        jnp.where(either_active, coin_flip, 0)
    )

    return mom._replace(
        weights=new_weights,
        active=new_active
    )


def genome_to_rendergraph(genome: Genome, n_input: int, n_output: int):
    """
    Convert a Python 'Genome' into the JSON structure that
    RenderGraph.drawGraph() expects.
    
    We'll build .nodes and .links, plus some constraints for alignment.
    """
    # 1) Mark all nodes as "active=1" for now (unless you want logic that disables certain nodes).
    n_nodes = genome.node_types.shape[0]
    # Build node objects: "name" (the type int), "active": 1
    # Because the RenderGraph code expects node.name to be an integer that indexes
    # into color tables (like 0=input, 1=output, 2=bias, etc.).
    node_list = []
    for i in range(n_nodes):
        # e.g. "name": node_type, "active": 1
        node_obj = {
            "name": int(genome.node_types[i]),
            "active": 1
        }
        node_list.append(node_obj)
    
    # 2) Build links array
    # We'll only include links if active[c] == 1
    links_list = []
    n_conn = genome.connections.shape[0]
    for c in range(n_conn):
        if genome.active[c] == 1:
            source_idx = int(genome.connections[c, 0])
            target_idx = int(genome.connections[c, 1])
            weight_val = float(genome.weights[c])
            links_list.append({
                "source": source_idx,
                "target": target_idx,
                "weight": weight_val
            })
    
    # 3) Build constraints array
    # Typically you want input nodes along one line, output nodes along another line, etc.
    # Suppose the first n_input are type=NODE_INPUT, then 1 bias, then n_output are output.
    # We'll create x/y alignments like the code does.
    # For simplicity, we’ll place input + bias near bottom, output near top.
    width_offsets = []
    height_offsets = []
    
    # input/bias: positions
    # We'll assume node indices [0..(n_input - 1)] are inputs, index n_input is bias,
    # then [n_input+1 .. n_input + n_output] are outputs, etc.
    for i in range(n_input + 1):
        width_offsets.append({"node": i, "offset": (i + 1) * 80})
        height_offsets.append({"node": i, "offset": 300})  # y=300 near bottom

    # output nodes
    for o in range(n_output):
        node_idx = n_input + 1 + o
        width_offsets.append({"node": node_idx, "offset": (o + 1) * 80})
        height_offsets.append({"node": node_idx, "offset": 60})  # near top

    constraints = [
        {
            "type": "alignment",
            "axis": "y",
            "offsets": height_offsets
        },
        {
            "type": "alignment",
            "axis": "x",
            "offsets": width_offsets
        }
    ]
    
    graph_obj = {
        "nodes": node_list,
        "links": links_list,
        "constraints": constraints
    }
    return graph_obj


######################################################
# DEMO USAGE
######################################################
def demo():
    rng = random.PRNGKey(42)

    # init
    n_input, n_output = 2, 1
    g = init_genome(rng, n_input, n_output, init_config="all")

    # fake data
    x = jnp.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype=jnp.float32)  # XOR example
    labels = jnp.array([0.0,1.0,1.0,0.0], dtype=jnp.float32)

    # forward
    out_before = forward(g, x)
    print("Output before backprop:\n", out_before)

    # do some backprop
    rng, _ = random.split(rng)
    g2 = backprop_genome(rng, g, x, labels, n_steps=100, lr=0.05)

    # forward after
    out_after = forward(g2, x)
    print("Output after  backprop:\n", out_after)


if __name__ == "__main__":
    demo()
