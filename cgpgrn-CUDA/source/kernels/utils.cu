#ifndef CGP_UTILS_CU
#define CGP_UTILS_CU

#include "../constants.h"

typedef struct {
    int topIndex;
    unsigned int info[MAX_NODES * MAX_ARITY];
} Stack;

typedef struct {
    int topIndex;
    float info[MAX_NODES * MAX_ARITY];
} ExStack;

typedef struct {
    int topIndex;
    float info[MAX_ARITY];
} ExStackLinear;

// ============================================================================
// OPERAÇÕES DE PILHA (STACK)
// ============================================================================

__host__ __device__ inline void push(Stack* s, unsigned int info) {
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY) {
        s->info[s->topIndex] = info;
    }
}

__host__ __device__ inline unsigned int pop(Stack* s) {
    if(s->topIndex >= 0) {
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
    return 0;
}

__host__ __device__ inline void pushEx(ExStack* s, float info) {
    (s->topIndex)++;
    if(s->topIndex < MAX_NODES * MAX_ARITY) {
        s->info[s->topIndex] = info;
    }
}

__host__ __device__ inline float popEx(ExStack* s) {
    if(s->topIndex >= 0) {
        (s->topIndex)--;
        return s->info[(s->topIndex) + 1];
    }
    return 0.0f;
}

__host__ __device__ inline void pushExLinear(ExStackLinear* s, float info) {
    (s->topIndex)++;
    s->info[s->topIndex] = info;
}

__host__ __device__ inline float popExLinear(ExStackLinear* s) {
    (s->topIndex)--;
    return s->info[(s->topIndex) + 1];
}

// ============================================================================
// GERAÇÃO DE NÚMEROS ALEATÓRIOS (LCG)
// ============================================================================

__host__ __device__ inline int rand2(int *seed) {
    int s = *seed;
    s = ((unsigned int)(s * 16807) % 2147483647);
    *seed = s;
    return s;
}

__host__ __device__ inline unsigned int randomInput(unsigned int index, int *seed) {
    return (rand2(seed) % (N + index));
}

__host__ __device__ inline unsigned int randomOutputIndex(int* seed) {
    return (rand2(seed) % MAX_NODES);
}

__host__ __device__ inline unsigned int randomFunction(int *seed) {
    return (rand2(seed) % (NUM_FUNCTIONS));
}

__host__ __device__ inline float randomConnectionWeight(int *seed) {
    return (((float) rand2(seed) / (float) (2147483647) ) * 2 * WEIGTH_RANGE) - WEIGTH_RANGE;
}

__host__ __device__ inline int randomInterval(int inf_bound, int sup_bound, int *seed) {
    return rand2(seed) % (sup_bound - inf_bound + 1) + inf_bound;
}

__host__ __device__ inline float randomProb(int* seed) {
    return (float)rand2(seed) / 2147483647.0f;
}

// ============================================================================
// MAPEAMENTO DE FUNÇÕES
// ============================================================================

__host__ __device__ inline unsigned int getFunctionInputs(unsigned int function) {
    switch (function) {
        #ifdef ADD
        case ADD:
        #endif
        #ifdef SUB
        case SUB:
        #endif
        #ifdef MUL
        case MUL:
        #endif
        #ifdef DIV
        case DIV:
        #endif
        #ifdef AND
        case AND:
        #endif
        #ifdef OR
        case OR:
        #endif
        #ifdef XOR
        case XOR:
        #endif
        #ifdef NAND
        case NAND:
        #endif
        #ifdef NOR
        case NOR:
        #endif
        #ifdef XNOR
        case XNOR:
        #endif
        #ifdef SIG
        case SIG:
        #endif
        #ifdef GAUSS
        case GAUSS:
        #endif
        #ifdef STEP
        case STEP:
        #endif
        #ifdef SOFTSIGN
        case SOFTSIGN:
        #endif
        #ifdef TANH
        case TANH:
            return MAX_ARITY;
        #endif
        #ifdef RAND
        case RAND:
        #endif
        #ifdef PI
        case PI:
        #endif
        #ifdef ONE
        case ONE:
        #endif
        #ifdef ZERO
        case ZERO:
            return 0;
        #endif
        #ifdef ABS
        case ABS:
        #endif
        #ifdef SQRT
        case SQRT:
        #endif
        #ifdef SQ
        case SQ:
        #endif
        #ifdef CUBE
        case CUBE:
        #endif
        #ifdef EXP
        case EXP:
        #endif
        #ifdef SIN
        case SIN:
        #endif
        #ifdef COS
        case COS:
        #endif
        #ifdef TAN
        case TAN:
        #endif
        #ifdef NOT
        case NOT:
        #endif
        #ifdef WIRE
        case WIRE:
            return 1;
        #endif
        #ifdef POW
        case POW:
            return 2;
        #endif
        default:
            return 0;
    }
}

#endif // CGP_UTILS_CU
