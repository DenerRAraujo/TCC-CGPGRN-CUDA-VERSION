

// NOTA: Certifique-se de que estruturas como Chromosome, Stack e as funções
// auxiliares (randomProb, push, pop, etc.) também estejam marcadas com __device__.

__device__ void activateNodes(Chromosome* c) {
    int i, j;
    int alreadyEvaluated[MAX_NODES];
    for(i = 0; i < MAX_NODES; i++) {
        alreadyEvaluated[i] = -1;
        c->activeNodes[i] = MAX_NODES + 1;
        c->nodes[i].active = 0;
    }

    c->numActiveNodes = 0;
    Stack s;
    s.topIndex = -1;

    for(i = 0; i < MAX_OUTPUTS; i++) {
        unsigned int nodeIndex = c->output[i];
        push(&s, nodeIndex);

        while(s.topIndex != -1) {
            unsigned int node = pop(&s);
            if(c->nodes[node].active == 0) {
                for (j = 0; j < MAX_ARITY; j++) {
                    if (c->nodes[node].inputs[j] >= N) {
                        push(&s, c->nodes[node].inputs[j] - N);
                    }
                }
                c->activeNodes[c->numActiveNodes] = node;
                c->nodes[node].active = 1;
                c->numActiveNodes++;
            }
        }
    }
}

__device__ void mutateTopologyProbabilistic(Chromosome *c, unsigned int* functionSet, int *seed, int type) {
    int i, j;
    // Em CUDA: get_local_id(0) vira threadIdx.x
    int local_id = threadIdx.x;

    for(i = 0; i < MAX_NODES; i++) {
        if(randomProb(seed) <= PROB_MUT) {
            c->nodes[i].function = functionSet[randomFunction(seed)];
            c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
        }

        for(j = 0; j < c->nodes[i].maxInputs; j++) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputs[j] = randomInput(i, seed);
            }
            if(type == 0 && randomProb(seed) <= PROB_MUT) {
                c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
            }
        }
    }

    activateNodes(c);
}

__device__ void mutateTopologyProbabilistic2(Chromosome *c, unsigned int* functionSet, int *seeds, int type) {
    int k;
    // Mapeamento de IDs do OpenCL para CUDA
    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    int size = MAX_NODES * MAX_ARITY;

    for(k = 0; k < ceil((size) / (float)LOCAL_SIZE_EVOL); k++) {
        int index = k * LOCAL_SIZE_EVOL + local_id;

        if(index < size) {
            int indexNode = index / MAX_ARITY;
            int indexArray = (index % MAX_ARITY);

            if(randomProb(seeds) <= PROB_MUT) {
                c->nodes[indexNode].inputs[indexArray] = randomInput(indexNode, seeds);
            }
            if(type == 0 && randomProb(seeds) <= PROB_MUT) {
                c->nodes[indexNode].inputsWeight[indexArray] = randomConnectionWeight(seeds);
            }

            if(indexArray == 0) {
                if(randomProb(seeds) <= PROB_MUT) {
                    c->nodes[indexNode].function = functionSet[randomFunction(seeds)];
                    c->nodes[indexNode].maxInputs = getFunctionInputs(c->nodes[indexNode].function);
                }
            }
        }
    }

    // Em CUDA: barrier(CLK_GLOBAL_MEM_FENCE) vira __syncthreads()
    __syncthreads();
}

__device__ void mutateTopologyProbabilisticActive(Chromosome *c, unsigned int* functionSet, int *seed, int type) {
    int i, j;
    for(i = 0; i < MAX_NODES; i++) {
        if(c->nodes[i].active == 1) {
            if(randomProb(seed) <= PROB_MUT) {
                c->nodes[i].function = functionSet[randomFunction(seed)];
                c->nodes[i].maxInputs = getFunctionInputs(c->nodes[i].function);
            }
            for(j = 0; j < c->nodes[i].maxInputs; j++) {
                if(randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputs[j] = randomInput(i, seed);
                }
                if(type == 0 && randomProb(seed) <= PROB_MUT) {
                    c->nodes[i].inputsWeight[j] = randomConnectionWeight(seed);
                }
            }
        }
    }

    activateNodes(c);
}

__device__ void mutateTopologyPoint(Chromosome *c, unsigned int* functionSet, int *seed) {
    int mutationComplete = -1;
    unsigned int newIndex;
    unsigned int newInputIndex;
    unsigned int newValue;

    int num_inputs = MAX_NODES * MAX_ARITY;
    while (mutationComplete == -1) {
        unsigned int nodeIndex = randomInterval(0, MAX_NODES + (num_inputs) + MAX_OUTPUTS, seed);

        if(nodeIndex < MAX_NODES) {
            newIndex = nodeIndex;
            newValue = functionSet[randomFunction(seed)];
            if(newValue != c->nodes[newIndex].function) {
                c->nodes[newIndex].function = newValue;
                c->nodes[newIndex].maxInputs = getFunctionInputs(newValue);
                if(c->nodes[newIndex].active > -1) {
                    mutationComplete = 1;
                }
            }
        } else if (nodeIndex <= MAX_NODES + (num_inputs)) {
            newIndex = (unsigned int) ((nodeIndex - MAX_NODES) / MAX_ARITY);
            newInputIndex = (unsigned int) ((nodeIndex - MAX_NODES) % MAX_ARITY);

            newValue = randomInput(newIndex, seed);

            if(newValue != c->nodes[newIndex].inputs[newInputIndex]) {
                c->nodes[newIndex].inputs[newInputIndex] = newValue;
                if(c->nodes[newIndex].active == 1) {
                    mutationComplete = 1;
                }
            }
        } else {
            newIndex = nodeIndex - (MAX_NODES + (num_inputs)) - 1;
            newValue = randomOutputIndex(seed);

            if(newValue != c->output[newIndex]) {
                c->output[newIndex] = newValue;
                mutationComplete = 1;
            }
        }
    }
    activateNodes(c);
}
