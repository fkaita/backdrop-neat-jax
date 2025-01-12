#!/usr/bin/env python
"""
A Python conversion of a NEAT implementation using JAX.
This code re-implements core functionality for Genome and NEATTrainer and
starts a simple Flask server to interact with them.
 
Reference: Original JavaScript version by David Ha (OToro Labs)

NOTE: This is a proof-of-concept re‑implementation. Many details of the recurrent
network propagation and clustering (k-medoids) have been simplified.
"""

import json
import math
import functools
from flask import Flask, request, jsonify
import jax.numpy as jnp
from jax import random
import jax  # for grad, jit, lax, etc.
import numpy as np  # for some Python list copies

# ---------------------------
# Global random key management
# ---------------------------
global_key = random.PRNGKey(0)

def randi(low, high):
    global global_key
    global_key, subkey = random.split(global_key)
    return int(random.randint(subkey, shape=(), minval=low, maxval=high))

def randn(mu, stdev):
    global global_key
    global_key, subkey = random.split(global_key)
    return float(mu + stdev * random.normal(subkey, shape=()))

def zeros(n):
    return jnp.zeros(n)


# ---------------------------
# Global constants and variables
# ---------------------------
# Node type constants
NODE_INPUT    = 0
NODE_OUTPUT   = 1
NODE_BIAS     = 2
NODE_SIGMOID  = 3
NODE_TANH     = 4
NODE_RELU     = 5
NODE_GAUSSIAN = 6
NODE_SIN      = 7
NODE_COS      = 8
NODE_ABS      = 9
NODE_MULT     = 10
NODE_ADD      = 11
NODE_MGAUSSIAN= 12
NODE_SQUARE   = 13

# For connections:
IDX_CONNECTION = 0
IDX_WEIGHT     = 1
IDX_ACTIVE     = 2

MAX_TICK = 100

# Operator mapping – names refer to functions implemented below
operators = [None, None, None, 'sigmoid', 'tanh', 'relu', 'gaussian', 'sin', 'cos', 'abs', 'mult', 'add', 'mult', 'add']

# initial configuration for connectivity ("none", "one", or "all")
initConfig = "none"

# Activation sets
activations_default = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_MULT, NODE_ADD] 
activations_all = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE]
activations_minimal = [NODE_RELU, NODE_TANH, NODE_GAUSSIAN, NODE_ADD]
activations = activations_default

def getRandomActivation():
    ix = randi(0, len(activations))
    return activations[ix]

gid = 0
def getGID():
    global gid
    result = gid
    gid += 1
    return result

# Global containers (these mimic the globals in the JS version)
nodes = []         # list of node types, e.g. NODE_INPUT, NODE_OUTPUT, etc.
connections = []   # global list of connections (each is a list: [from_node, to_node])

def copyArray(x):
    return x.copy()

def copyConnections(newC):
    # returns a copy of the connections (only first two elements per connection)
    return [[c[0], c[1]] for c in newC]

def getNodes():
    return copyArray(nodes)

def getConnections():
    return copyConnections(connections)

# Global network configuration variables
nInput  = 1
nOutput = 1
outputIndex = 2   # (bias, then inputs, then outputs)
generationNum = 0

def incrementGenerationCounter():
    global generationNum
    generationNum += 1

# Simple render mode functions (for display in the original code)
def getRandomRenderMode():
    z = randi(0, 6)
    if z < 3:
        return 0
    if z < 5:
        return 1
    return 2

renderMode = getRandomRenderMode()

def randomizeRenderMode():
    global renderMode
    renderMode = getRandomRenderMode()
    print(f'render mode = {renderMode}')

def setRenderMode(rMode):
    global renderMode
    renderMode = rMode

def getRenderMode():
    return renderMode

def getOption(opt, key, default):
    if opt and key in opt:
        return opt[key]
    return default

def init(opt=None):
    """Initializes the global network variables."""
    global nInput, nOutput, initConfig, nodes, connections, outputIndex, generationNum, activations
    opt = opt or {}
    nInput  = getOption(opt, 'nInput', nInput)
    nOutput = getOption(opt, 'nOutput', nOutput)
    initConfig = getOption(opt, 'initConfig', initConfig)
    if 'activations' in opt:
        if opt['activations'] == "all":
            activations = activations_all
        elif opt['activations'] == "minimal":
            activations = activations_minimal
    outputIndex = nInput + 1  # index for first output (after bias and inputs)
    nodes = []
    connections = []
    generationNum = 0
    # initialize nodes: inputs then bias then outputs
    for i in range(nInput):
        nodes.append(NODE_INPUT)
    nodes.append(NODE_BIAS)
    for i in range(nOutput):
        nodes.append(NODE_OUTPUT)
    # initialize connections – by default connect inputs to outputs if config is "all"
    if initConfig == "all":
        for j in range(nOutput):
            for i in range(nInput+1):
                connections.append([i, outputIndex+j])
    elif initConfig == "one":
        # add a dummy hidden node
        nodes.append(NODE_ADD)
        dummyIndex = len(nodes) - 1
        for i in range(nInput+1):
            connections.append([i, dummyIndex])
        for i in range(nOutput):
            connections.append([dummyIndex, outputIndex+i])
    # (for "none", no connections are added globally)

# ---------------------------
# Simple neural operations (using JAX)
# ---------------------------
class GraphOps:
    """A set of operations that mimic the recurrent.js operations."""
    
    @staticmethod
    def mul(a, b):
        # elementwise multiplication (assume a and b are scalars or 1-element arrays)
        return a * b

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def eltmul(a, b):
        # elementwise multiplication (identical to mul in our scalar case)
        return a * b

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    @staticmethod
    def tanh(x):
        return jnp.tanh(x)

    @staticmethod
    def relu(x):
        return jnp.maximum(0, x)

    @staticmethod
    def gaussian(x):
        return jnp.exp(-x*x)

    @staticmethod
    def sin(x):
        return jnp.sin(x)

    @staticmethod
    def cos(x):
        return jnp.cos(x)

    @staticmethod
    def abs(x):
        return jnp.abs(x)

# ---------------------------
# Genome class
# ---------------------------
class Genome:
    def __init__(self, initGenome=None):
        self.connections = []  # local copy of (global) connections; each is [innovation, weight, active]
        # If an initial genome is provided, copy its connections.
        if initGenome is not None and hasattr(initGenome, 'connections'):
            for c in initGenome.connections:
                self.connections.append(c.copy())
        else:
            # Initialize based on global initConfig
            if initConfig == "all":
                for i in range((nInput+1)*nOutput):
                    # create a connection using a list of three values
                    c = [i, randn(0.0, 1.0), 1]
                    self.connections.append(c)
            elif initConfig == "one":
                total = (nInput+1) + nOutput
                for i in range(total):
                    c = [i, randn(0.0, 1.0) if i < (nInput+1) else 1.0, 1]
                    self.connections.append(c)
        self.fitness = -1e20
        self.cluster = 0
        self.unrolledConnections = None  # will be created later

    def copy(self):
        g = Genome(self)
        g.fitness = self.fitness
        g.cluster = self.cluster
        return g

    def importConnections(self, cArray):
        self.connections = []
        for c in cArray:
            self.connections.append([c[0], c[1], c[2]])
    
    def copyFrom(self, sourceGenome):
        self.importConnections(sourceGenome.connections)
        self.fitness = sourceGenome.fitness
        self.cluster = sourceGenome.cluster

    def mutateWeights(self, mutationRate_=None, mutationSize_=None):
        mRate = mutationRate_ if mutationRate_ is not None else 0.2
        mSize = mutationSize_ if mutationSize_ is not None else 0.5
        for i in range(len(self.connections)):
            if np.random.rand() < mRate:
                self.connections[i][IDX_WEIGHT] += randn(0, mSize)

    def areWeightsNaN(self):
        for c in self.connections:
            if math.isnan(c[IDX_WEIGHT]):
                return True
        return False

    def clipWeights(self, maxWeight_=50.0):
        maxW = abs(maxWeight_)
        for i in range(len(self.connections)):
            w = self.connections[i][IDX_WEIGHT]
            assert not math.isnan(w), "weight had NaN."
            w = min(maxW, w)
            w = max(-maxW, w)
            self.connections[i][IDX_WEIGHT] = w

    def getAllConnections(self):
        return connections

    def addRandomNode(self):
        if len(self.connections) == 0:
            return
        c_index = randi(0, len(self.connections))
        # Only proceed if chosen connection is active.
        if self.connections[c_index][IDX_ACTIVE] != 1:
            return
        w = self.connections[c_index][IDX_WEIGHT]
        self.connections[c_index][IDX_ACTIVE] = 0  # disable that connection
        nodeIndex = len(nodes)
        # add a new node with a random activation type
        nodes.append(getRandomActivation())
        # use the global connection indexed by innovation number (assume innovation equals same index)
        innovationNum = self.connections[c_index][IDX_CONNECTION]
        fromNodeIndex = connections[innovationNum][0]
        toNodeIndex = connections[innovationNum][1]
        connectionIndex = len(connections)
        # add two new global connections
        connections.append([fromNodeIndex, nodeIndex])
        connections.append([nodeIndex, toNodeIndex])
        # add local connections for this genome
        c1 = [connectionIndex, 1.0, 1]
        c2 = [connectionIndex+1, w, 1]
        self.connections.append(c1)
        self.connections.append(c2)

    def getNodesInUse(self):
        nNodes = len(nodes)
        nodesInUseFlag = [0] * nNodes
        for c in self.connections:
            global_innov = c[IDX_CONNECTION]
            from_idx, to_idx = connections[global_innov]
            nodesInUseFlag[from_idx] = 1
            nodesInUseFlag[to_idx] = 1
        nodesInUse = []
        for i in range(nNodes):
            # always include input, bias, output nodes (first nInput+1+nOutput)
            if nodesInUseFlag[i] == 1 or (i < nInput+1+nOutput):
                nodesInUse.append(i)
        return nodesInUse

    def addRandomConnection(self):
        nodesInUse = self.getNodesInUse()
        if len(nodesInUse) == 0:
            return
        # choose two different nodes (avoid output-to-output if possible)
        # (This is a simplified version compared to the original code.)
        fromNodeIndex = nodesInUse[randi(0, len(nodesInUse))]
        toNodeIndex = nodesInUse[randi(0, len(nodesInUse))]
        if fromNodeIndex == toNodeIndex:
            return
        # Check if connection already exists globally
        searchIndex = -1
        for i, con in enumerate(connections):
            if con[0] == fromNodeIndex and con[1] == toNodeIndex:
                searchIndex = i
                break
        if searchIndex < 0:
            connectionIndex = len(connections)
            connections.append([fromNodeIndex, toNodeIndex])
            c = [connectionIndex, randn(0.0, 1.0), 1]
            self.connections.append(c)
        else:
            # if it exists, enable it if it is not already in this genome.
            found = False
            for c in self.connections:
                if c[IDX_CONNECTION] == searchIndex:
                    if c[IDX_ACTIVE] == 0:
                        c[IDX_WEIGHT] = randn(0.0, 1.0)
                        c[IDX_ACTIVE] = 1
                    found = True
                    break
            if not found:
                c1 = [searchIndex, randn(0.0, 1.0), 1]
                self.connections.append(c1)

    def createUnrolledConnections(self):
        total = len(connections)
        self.unrolledConnections = [[0, 0.0, 0] for _ in range(total)]
        for c in self.connections:
            cIndex = c[IDX_CONNECTION]
            self.unrolledConnections[cIndex] = c.copy()

    def crossover(self, other):
        # Create an offspring genome by combining self and other.
        self.createUnrolledConnections()
        other.createUnrolledConnections()
        child = Genome()
        child.connections = []
        total = len(connections)
        for i in range(total):
            count = 0
            g = self
            if self.unrolledConnections[i][IDX_CONNECTION] == 1:
                count += 1
            if other.unrolledConnections[i][IDX_CONNECTION] == 1:
                g = other
                count += 1
            if count == 2 and np.random.rand() < 0.5:
                g = self
            if count == 0:
                continue
            c = [i, g.unrolledConnections[i][IDX_WEIGHT], 1]
            if (self.unrolledConnections[i][IDX_ACTIVE] == 0 and
                other.unrolledConnections[i][IDX_ACTIVE] == 0):
                c[IDX_ACTIVE] = 0
            child.connections.append(c)
        return child

    def roundWeights(self):
        precision = 10000
        for i in range(len(self.connections)):
            w = self.connections[i][IDX_WEIGHT]
            self.connections[i][IDX_WEIGHT] = round(w*precision)/precision

    def toJSON(self, description=""):
        data = {
            "nodes": copyArray(nodes),
            "connections": copyConnections(connections),
            "nInput": nInput,
            "nOutput": nOutput,
            "renderMode": renderMode,
            "outputIndex": outputIndex,
            "genome": self.connections,
            "description": description
        }
        return json.dumps(data)


    def fromJSON(self, data_string):
        data = json.loads(data_string)
        global nodes, connections, nInput, nOutput, renderMode, outputIndex
        nodes = data["nodes"].copy()
        connections = copyConnections(data["connections"])
        nInput = data["nInput"]
        nOutput = data["nOutput"]
        renderMode = data.get("renderMode", 0)
        outputIndex = data["outputIndex"]
        self.importConnections(data["genome"])
        return data.get("description", "")

    def forward(self, input_values=None, weights=None):
        """
        A differentiable forward propagation routine.

        Parameters:
          input_values (optional): A JAX array of inputs for the input nodes.
                                   If None, defaults to 0.5 for each input.
          weights (optional): A JAX array of connection weights.
                              If None, the method uses the weights stored in self.connections.

        Returns:
          A JAX array of outputs for the output nodes.
        """
        nNodes = len(nodes)
        # Convert the global nodes list into a JAX array.
        nodes_array = jnp.array(nodes)
        # Identify indices for input, bias, and output nodes.
        input_indices = jnp.array([i for i, nt in enumerate(nodes) if nt == NODE_INPUT])
        bias_indices  = jnp.array([i for i, nt in enumerate(nodes) if nt == NODE_BIAS])
        output_indices = jnp.array([i for i, nt in enumerate(nodes) if nt == NODE_OUTPUT])

        # Use the provided weights or extract them from the genome.
        if weights is None:
            weights = jnp.array([c[IDX_WEIGHT] for c in self.connections])
        # Always convert the "active" flag into a JAX array.
        active_mask = jnp.array([c[IDX_ACTIVE] for c in self.connections])
        # Get the list of global connection indices from this genome.
        conn_idx_list = [c[IDX_CONNECTION] for c in self.connections]
        # Convert the global connections list to a JAX array.
        global_conns = jnp.array(connections)  # shape: (total_connections, 2)
        # Pick out only the rows that this genome uses.
        selected = global_conns[conn_idx_list]  # shape: (# genome connections, 2)
        from_indices = selected[:, 0].astype(jnp.int32)
        to_indices   = selected[:, 1].astype(jnp.int32)

        # Default input values if none provided.
        if input_values is None:
            input_values = jnp.array([0.5] * nInput)
        # Initialize node values: use provided input_values for input nodes, bias nodes fixed at 1.0,
        # and zeros for all other nodes.
        node_vals = jnp.zeros(nNodes)
        node_vals = node_vals.at[input_indices].set(input_values)
        node_vals = node_vals.at[bias_indices].set(1.0)

        # Define a JAX-compatible update for one "tick" of propagation.
        def body_fn(t, nv):
            # Each connection contributes its weight * active_mask * the value of its "from" node.
            contrib = weights * active_mask * nv[from_indices]
            # Sum contributions per destination node.
            summed = jax.ops.segment_sum(contrib, to_indices, nNodes)
            # Apply the activation function for each node.
            def apply_activation(i, x):
                nt = nodes_array[i]
                return jnp.where(
                    nt == NODE_SIGMOID, GraphOps.sigmoid(x),
                    jnp.where(
                        nt == NODE_TANH, GraphOps.tanh(x),
                        jnp.where(
                            nt == NODE_RELU, GraphOps.relu(x),
                            jnp.where(
                                nt == NODE_GAUSSIAN, GraphOps.gaussian(x),
                                jnp.where(
                                    nt == NODE_SIN, GraphOps.sin(x),
                                    jnp.where(
                                        nt == NODE_MULT, GraphOps.mul(x),
                                        jnp.where(nt == NODE_ADD, GraphOps.add(x), x)
                                    )
                                )
                            )
                        )
                    )
                )
            # Vectorize the activation application over all nodes.
            new_nv = jax.vmap(apply_activation)(jnp.arange(nNodes), summed)
            return new_nv

        # Run a fixed number of update iterations.
        final_vals = jax.lax.fori_loop(0, MAX_TICK, body_fn, node_vals)
        # Return the outputs corresponding to the output nodes.
        return final_vals[output_indices]

    def backward(self, loss_fn, input_values=None):
        """
        Computes the gradient of a scalar loss with respect to the genome's connection weights.
        
        Parameters:
          loss_fn    : A function that accepts the network outputs (a JAX array)
                       and returns a scalar loss.
          input_values (optional): A JAX array to use as inputs (default is 0.5 for each input).
        
        Returns:
          loss_val    : The scalar loss computed using the current weights.
          grad_weights: A JAX array of gradients with respect to each connection weight.
        """
        # Extract the initial weights from the genome.
        weights_init = jnp.array([c[IDX_WEIGHT] for c in self.connections])
        # Define a loss function that calls the (differentiable) forward method.
        # The forward method accepts an optional weight vector.
        loss_from_weights = lambda w: loss_fn(self.forward(input_values=input_values, weights=w))
        # Compute the gradient of the loss with respect to the weights.
        grad_weights = jax.grad(loss_from_weights)(weights_init)
        # Also compute the current loss.
        loss_val = loss_from_weights(weights_init)
        return loss_val, grad_weights

# ---------------------------
# A simple k-medoid clustering stub
# ---------------------------
class KMedoids:
    def __init__(self):
        self.K = 0
        self.dist_func = None
        self.clusters = None

    def init(self, K):
        self.K = K

    def setDistFunction(self, f):
        self.dist_func = f

    def partition(self, gene_list):
        # A simple (and not very efficient) clustering: assign genes round-robin.
        K = self.K if self.K > 0 else 1
        self.clusters = [[] for _ in range(K)]
        for idx, gene in enumerate(gene_list):
            self.clusters[idx % K].append(idx)

    def getCluster(self):
        return self.clusters if self.clusters is not None else []

# ---------------------------
# NEATTrainer class
# ---------------------------
class NEATTrainer:
    def __init__(self, options=None, initGenome=None):
        opts = options or {}
        self.num_populations = opts.get("num_populations", 5)
        self.sub_population_size = opts.get("sub_population_size", 10)
        self.hall_of_fame_size = opts.get("hall_of_fame_size", 5)
        self.new_node_rate = opts.get("new_node_rate", 0.1)
        self.new_connection_rate = opts.get("new_connection_rate", 0.1)
        self.extinction_rate = opts.get("extinction_rate", 0.5)
        self.mutation_rate = opts.get("mutation_rate", 0.1)
        self.mutation_size = opts.get("mutation_size", 1.0)
        self.init_weight_magnitude = opts.get("init_weight_magnitude", 1.0)
        self.target_fitness = opts.get("target_fitness", 1e20)
        self.debug_mode = opts.get("debug_mode", False)
        self.forceExtinctionMode = False

        # Set globals used by Genome mutations:
        global mutationRate, mutationSize
        mutationRate = self.mutation_rate
        mutationSize = self.mutation_size

        N = self.sub_population_size
        K = self.num_populations
        self.kmedoids = KMedoids()
        self.kmedoids.init(K)
        self.kmedoids.setDistFunction(self.dist)

        self.genes = []
        self.hallOfFame = []
        self.bestOfSubPopulation = []

        # Create an initial population:
        for i in range(N*K):
            if initGenome is not None:
                genome = Genome(initGenome)
            else:
                genome = Genome()
            genome.addRandomConnection()
            genome.mutateWeights(1.0, self.mutation_size)  # burst mutate init weights
            genome.fitness = -1e20
            genome.cluster = randi(0, K)
            self.genes.append(genome)
        # initialize hall-of-fame
        for i in range(self.hall_of_fame_size):
            if initGenome is not None:
                genome = Genome(initGenome)
            else:
                genome = Genome()
                genome.addRandomConnection()
                genome.mutateWeights(1.0, self.mutation_size)
            genome.fitness = -1e20
            genome.cluster = 0
            self.hallOfFame.append(genome)

    def sortByFitness(self, gene_list):
        gene_list.sort(key=lambda g: g.fitness, reverse=True)

    def forceExtinction(self):
        self.forceExtinctionMode = True

    def resetForceExtinction(self):
        self.forceExtinctionMode = False

    def applyMutations(self, g):
        if np.random.rand() < self.new_node_rate:
            g.addRandomNode()
        if np.random.rand() < self.new_connection_rate:
            g.addRandomConnection()
        g.mutateWeights(self.mutation_rate, self.mutation_size)

    def applyFitnessFuncToList(self, f, geneList):
        for g in geneList:
            g.fitness = f(g)

    def getAllGenes(self):
        return self.genes + self.hallOfFame + self.bestOfSubPopulation

    def applyFitnessFunc(self, fitness_func_code, clusterMode=True):
        # Deserialize the fitness function (assumes it's Python code as a string)
        local_context = {}
        exec(fitness_func_code, globals(), local_context)
        fitness_func = local_context.get("fitness_func")
        if not fitness_func:
            raise ValueError("Invalid fitness function provided.")

        # Apply the fitness function
        self.applyFitnessFuncToList(fitness_func, self.genes)
        self.applyFitnessFuncToList(fitness_func, self.hallOfFame)
        self.applyFitnessFuncToList(fitness_func, self.bestOfSubPopulation)
        self.filterFitness()
        combined = self.genes + self.hallOfFame + self.bestOfSubPopulation
        self.sortByFitness(combined)
        if clusterMode:
            self.cluster()
        # Update hall-of-fame
        self.hallOfFame = [combined[i].copy() for i in range(self.hall_of_fame_size)]
        K = self.num_populations
        self.bestOfSubPopulation = []
        for j in range(K):
            for g in combined:
                if g.cluster == j:
                    self.bestOfSubPopulation.append(g.copy())
                    break


    def clipWeights(self, maxWeight_=50.0):
        for g in self.genes:
            g.clipWeights(maxWeight_)
        for g in self.hallOfFame:
            g.clipWeights(maxWeight_)

    def areWeightsNaN(self):
        for g in self.genes + self.hallOfFame:
            if g.areWeightsNaN():
                return True
        return False

    def filterFitness(self):
        epsilon = 1e-10
        def process(g):
            fitness = -1e20
            if g.fitness is not None and not math.isnan(g.fitness):
                fitness = -abs(g.fitness)
                fitness = min(fitness, -epsilon)
            g.fitness = fitness
        for g in self.genes + self.hallOfFame:
            process(g)

    def pickRandomIndex(self, genes, cluster=None):
        totalProb = 0.0
        slack = 0.01
        normFitness = []
        eligible = []
        for g in genes:
            if cluster is None or g.cluster == cluster:
                val = 1.0 / (-g.fitness + slack)
                normFitness.append(val)
                eligible.append(g)
                totalProb += val
        if not eligible:
            return None
        normFitness = [v/totalProb for v in normFitness]
        x = np.random.rand()
        for idx, prob in enumerate(normFitness):
            x -= prob
            if x <= 0:
                return idx
        return len(eligible)-1

    def cluster(self, genePool=None):
        genePool = self.genes if genePool is None else genePool
        self.kmedoids.partition(genePool)
        clusters = self.kmedoids.getCluster()
        for cluster_idx, indices in enumerate(clusters):
            for idx in indices:
                genePool[idx].cluster = cluster_idx

    def evolve(self, mutateWeightsOnly=False):
        prevGenes = self.genes
        newGenes = []
        K = self.num_populations
        N = self.sub_population_size
        incrementGenerationCounter()
        self.kmedoids.partition(prevGenes)
        clusters = self.kmedoids.getCluster()
        worstFitness = 1e20
        worstCluster = -1
        bestFitness = -1e20
        bestCluster = -1
        for i in range(K):
            if len(clusters[i]) == 0:
                continue
            # sort indices by fitness (highest first)
            indices = clusters[i]
            sorted_genes = sorted([prevGenes[j] for j in indices], key=lambda g: g.fitness, reverse=True)
            if sorted_genes[0].fitness < worstFitness:
                worstFitness = sorted_genes[0].fitness
                worstCluster = i
            if sorted_genes[0].fitness >= bestFitness:
                bestFitness = sorted_genes[0].fitness
                bestCluster = i
        extinctionEvent = False
        if np.random.rand() < self.extinction_rate and not mutateWeightsOnly:
            extinctionEvent = True
            if self.debug_mode:
                print('Crappiest sub-population will be extinct.')
        if self.forceExtinctionMode and not mutateWeightsOnly:
            extinctionEvent = True
            if self.debug_mode:
                print('Forced extinction.')
        for i in range(K):
            for j in range(N):
                if extinctionEvent and i == worstCluster:
                    idx_mom = self.pickRandomIndex(prevGenes, bestCluster)
                    idx_dad = self.pickRandomIndex(prevGenes, bestCluster)
                else:
                    idx_mom = self.pickRandomIndex(prevGenes, i)
                    idx_dad = self.pickRandomIndex(prevGenes, i)
                try:
                    mom = prevGenes[idx_mom]
                    dad = prevGenes[idx_dad]
                    if mutateWeightsOnly:
                        baby = mom.crossover(dad)
                        baby.mutateWeights(self.mutation_rate, self.mutation_size)
                    else:
                        baby = mom.crossover(dad)
                        self.applyMutations(baby)
                except Exception as err:
                    if self.debug_mode:
                        print("Error during crossover:", err)
                    baby = prevGenes[0].copy()
                    self.applyMutations(baby)
                baby.cluster = i
                newGenes.append(baby)
        self.genes = newGenes
        # (Compression step is omitted for brevity.)
    
    def printFitness(self):
        for i, g in enumerate(self.genes):
            print(f"Genome {i} fitness = {g.fitness}")
        for i, g in enumerate(self.hallOfFame):
            print(f"HallOfFamer {i} fitness = {g.fitness}")
        for i, g in enumerate(self.bestOfSubPopulation):
            print(f"BestOfSubPopulation {i} fitness = {g.fitness}")

    def getBestGenome(self, cluster=None):
        allGenes = self.genes.copy()
        self.sortByFitness(allGenes)
        if cluster is None:
            return allGenes[0]
        for g in allGenes:
            if g.cluster == cluster:
                return g
        return allGenes[0]
    
    # Add a helper method to convert the best genome to JSON
    def getBestGenomeJSON(self, cluster=None):
        best_genome = self.getBestGenome(cluster)
        return best_genome.toJSON()

    def dist(self, g1, g2):
        g1.createUnrolledConnections()
        g2.createUnrolledConnections()
        coef = {"excess": 10.0, "disjoint": 10.0, "weight": 0.1}
        nBothActive = 0
        nDisjoint = 0
        nExcess = 0
        weightDiff = 0.0
        lastIndex1 = -1
        lastIndex2 = -1
        total = len(connections)
        diffVector = [0]*total
        for i in range(total):
            c1 = g1.unrolledConnections[i]
            c2 = g2.unrolledConnections[i]
            exist1 = c1[IDX_CONNECTION]
            exist2 = c2[IDX_CONNECTION]
            active1 = exist1 * c1[IDX_ACTIVE]
            active2 = exist2 * c2[IDX_ACTIVE]
            if exist1 == 1:
                lastIndex1 = i
            if exist2 == 1:
                lastIndex2 = i
            diffVector[i] = 0 if exist1 == exist2 else 1
            if active1 == 1 and active2 == 1:
                w1 = c1[IDX_WEIGHT]
                w2 = c2[IDX_WEIGHT]
                nBothActive += 1
                weightDiff += abs(w1 - w2)
        minIndex = min(lastIndex1 if lastIndex1>=0 else 0, lastIndex2 if lastIndex2>=0 else 0)
        if nBothActive > 0:
            weightDiff /= nBothActive
        for i in range(minIndex+1):
            nDisjoint += diffVector[i]
        for i in range(minIndex+1, total):
            nExcess += diffVector[i]
        numNodes = max(len(self.getBestGenome().getNodesInUse()), len(self.getBestGenome().getNodesInUse()))
        distDisjoint = coef["disjoint"] * nDisjoint / (numNodes if numNodes>0 else 1)
        distExcess = coef["excess"] * nExcess / (numNodes if numNodes>0 else 1)
        distWeight = coef["weight"] * weightDiff
        distance = distDisjoint + distExcess + distWeight
        if math.isnan(distance) or abs(distance) > 100:
            print("large distance report:")
            print("distance =", distance)
        return distance
