#include <cstdio>
#include "../constants.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <float.h>




__constant__ unsigned int d_N;
__constant__ unsigned int d_M;
__constant__ unsigned int d_M_TRAIN;
__constant__ unsigned int d_M_VALIDATION;
__constant__ unsigned int d_M_TEST;
__constant__ unsigned int d_NUM_FUNCTIONS;
__constant__ float d_WEIGTH_RANGE;

__constant__ int d_LOCAL_SIZE;
__constant__ int d_LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2;
__constant__ int d_LOCAL_SIZE_TRAIN;
__constant__ int d_LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2;
__constant__ int d_LOCAL_SIZE_VALIDATION;
__constant__ int d_LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2;
__constant__ int d_LOCAL_SIZE_TEST;
__constant__ int d_LOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2;
__constant__ int d_LOCAL_SIZE_EVOL;


#include "utils.cu"
#include "evol.cu"

// ============================================================================
// FUNÇÕES AUXILIARES NA GPU
// ============================================================================

static __device__ float executeFunction(Chromosome* c, int node, ExStack* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    unsigned int inputs = c->nodes[node].maxInputs;

    switch (c->nodes[node].function) {
        #ifdef ADD
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabsf(popEx(exStack));
            break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrtf(popEx(exStack));
            break;
        #endif

        #ifdef SQ
        case SQ:
            result = powf(popEx(exStack), 2.0f);
            break;
        #endif

        #ifdef CUBE
        case CUBE:
            result = powf(popEx(exStack), 3.0f);
            break;
        #endif

        #ifdef POW
        case POW:
            result = popEx(exStack);
            result = powf(popEx(exStack), result);
            break;
        #endif

        #ifdef AND
        case AND:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 0.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef OR
        case OR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 1.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef XOR
        case XOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result != 1.0f) {
                result = 0.0f;
            }
            break;
        #endif

        #ifdef NAND
        case NAND:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 0.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOT
        case NOT:
            for (i = 0; i < 1; i++) {
                if (popEx(exStack) == 1.0f) { 
                    result = 0.0f;
                } else {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOR
        case NOR:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 1.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef XNOR
        case XNOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popEx(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result == 1.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef EXP
        case EXP:
            result = expf(popEx(exStack));
            break;
        #endif

        #ifdef SIN
        case SIN:
            result = sinf(popEx(exStack));
            break;
        #endif

        #ifdef COS
        case COS:
            result = cosf(popEx(exStack));
            break;
        #endif

        #ifdef TAN
        case TAN:
            result = tanf(popEx(exStack));
            break;
        #endif

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popEx(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popEx(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanhf(sum);
            break;
        #endif

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinear(Chromosome* c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    unsigned int inputs = c->nodes[node].maxInputs;

    switch (c->nodes[node].function) {
        #ifdef ADD
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabsf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrtf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQ
        case SQ:
            result = powf(popExLinear(exStack), 2.0f);
            break;
        #endif

        #ifdef CUBE
        case CUBE:
            result = powf(popExLinear(exStack), 3.0f);
            break;
        #endif

        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = powf(popExLinear(exStack), result);
            break;
        #endif

        #ifdef AND
        case AND:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef OR
        case OR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOT
        case NOT:
            for (i = 0; i < 1; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 0.0f;
                } else {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef XOR
        case XOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result != 1.0f) {
                result = 0.0f;
            }
            break;
        #endif

        #ifdef NAND
        case NAND:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOR
        case NOR:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef XNOR
        case XNOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result == 1.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef EXP
        case EXP:
            result = expf(popExLinear(exStack));
            break;
        #endif

        #ifdef SIN
        case SIN:
            result = sinf(popExLinear(exStack));
            break;
        #endif

        #ifdef COS
        case COS:
            result = cosf(popExLinear(exStack));
            break;
        #endif

        #ifdef TAN
        case TAN:
            result = tanf(popExLinear(exStack));
            break;
        #endif

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanhf(sum);
            break;
        #endif

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearCompact(CompactChromosome* c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    unsigned int aux;
    unsigned int inputs = MAX_ARITY; // c->nodes[node].maxInputs;

    // printf("\n%d", c->nodes[node].function_inputs_active);
    switch (c->nodes[node].function_inputs_active >> 17) {

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                // aux = c->nodes[node].inputsWeight[i] & (0xFFFF0000);
                // printf("\n%f", (*(float*)(&aux)));

                // sum += (popEx(exStack) * (*(float*)(&aux)));

                // aux = (c->nodes[node].inputsWeight[i] << 16);
                // printf("\n%f", int_float.f);

                // sum += (popEx(exStack) * (*(float*)(&aux)));
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearActive(ActiveChromosome* c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    unsigned int inputs = c->nodes[node].maxInputs;

    switch (c->nodes[node].function) {
        #ifdef ADD
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabsf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrtf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQ
        case SQ:
            result = powf(popExLinear(exStack), 2.0f);
            break;
        #endif

        #ifdef CUBE
        case CUBE:
            result = powf(popExLinear(exStack), 3.0f);
            break;
        #endif

        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = powf(popExLinear(exStack), result);
            break;
        #endif

        #ifdef AND
        case AND:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef OR
        case OR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef XOR
        case XOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result != 1.0f) {
                result = 0.0f;
            }
            break;
        #endif

        #ifdef NAND
        case NAND:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOR
        case NOR:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef XNOR
        case XNOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result == 1.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef EXP
        case EXP:
            result = expf(popExLinear(exStack));
            break;
        #endif

        #ifdef SIN
        case SIN:
            result = sinf(popExLinear(exStack));
            break;
        #endif

        #ifdef COS
        case COS:
            result = cosf(popExLinear(exStack));
            break;
        #endif

        #ifdef TAN
        case TAN:
            result = tanf(popExLinear(exStack));
            break;
        #endif

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                sum += (popExLinear(exStack) * c->nodes[node].inputsWeight[i]);
            }
            result = tanhf(sum);
            break;
        #endif

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearActiveImage(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;

    // Equivalente ao get_group_id(0) do OpenCL
    int group_id = blockIdx.x;

    // Lê o pixel com tipo inteiro no CUDA (substitui uint4 e read_imageui)
    // Coordenadas tex3D: x, y, z (aqui representam: coluna, linha, camada/array)
    uint4 pixelInt = tex3D<uint4>(c, 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
    float4 pixelFloat;

    unsigned int inputs = MAX_ARITY; // c->nodes[node].maxInputs;

    switch (pixelInt.x) {
        #ifdef ADD
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabsf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrtf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQ
        case SQ:
            result = powf(popExLinear(exStack), 2.0f);
            break;
        #endif

        #ifdef CUBE
        case CUBE:
            result = powf(popExLinear(exStack), 3.0f);
            break;
        #endif

        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = powf(popExLinear(exStack), result);
            break;
        #endif

        #ifdef AND
        case AND:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef OR
        case OR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef XOR
        case XOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result != 1.0f) {
                result = 0.0f;
            }
            break;
        #endif

        #ifdef NAND
        case NAND:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOR
        case NOR:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef XNOR
        case XNOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result == 1.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef EXP
        case EXP:
            result = expf(popExLinear(exStack));
            break;
        #endif

        #ifdef SIN
        case SIN:
            result = sinf(popExLinear(exStack));
            break;
        #endif

        #ifdef COS
        case COS:
            result = cosf(popExLinear(exStack));
            break;
        #endif

        #ifdef TAN
        case TAN:
            result = tanf(popExLinear(exStack));
            break;
        #endif

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = tanhf(sum);
            break;
        #endif

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearActiveImageHalf(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;

    // Mapeamento do ID do bloco no CUDA (equivalente a get_group_id(0))
    int group_id = blockIdx.x;

    // Leitura do tipo da função no primeiro pixel (0, node+1, group_id)
    uint4 pixelInt = tex3D<uint4>(c, 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
    float4 pixelFloat;

    unsigned int inputs = MAX_ARITY; // c->nodes[node].maxInputs;

    switch (pixelInt.x) {
        #ifdef ADD
        case ADD:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result += exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef SUB
        case SUB:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result -= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef MUL
        case MUL:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result *= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef DIV
        case DIV:
            result = exStack->info[exStack->topIndex - inputs + 1];
            for (i = 1; i < inputs; i++) {
                result /= exStack->info[exStack->topIndex - i + 1];
            }
            exStack->topIndex -= inputs;
            break;
        #endif

        #ifdef ABS
        case ABS:
            result = fabsf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQRT
        case SQRT:
            result = sqrtf(popExLinear(exStack));
            break;
        #endif

        #ifdef SQ
        case SQ:
            result = powf(popExLinear(exStack), 2.0f);
            break;
        #endif

        #ifdef CUBE
        case CUBE:
            result = powf(popExLinear(exStack), 3.0f);
            break;
        #endif

        #ifdef POW
        case POW:
            result = popExLinear(exStack);
            result = powf(popExLinear(exStack), result);
            break;
        #endif

        #ifdef AND
        case AND:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef OR
        case OR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef XOR
        case XOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result != 1.0f) {
                result = 0.0f;
            }
            break;
        #endif

        #ifdef NAND
        case NAND:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 0.0f) {
                    result = 1.0f;
                }
            }
            break;
        #endif

        #ifdef NOR
        case NOR:
            result = 1.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result = 0.0f;
                }
            }
            break;
        #endif

        #ifdef XNOR
        case XNOR:
            result = 0.0f;
            for (i = 0; i < inputs; i++) {
                if (popExLinear(exStack) == 1.0f) {
                    result += 1.0f;
                }
            }
            if (result == 1.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef EXP
        case EXP:
            result = expf(popExLinear(exStack));
            break;
        #endif

        #ifdef SIN
        case SIN:
            result = sinf(popExLinear(exStack));
            break;
        #endif

        #ifdef COS
        case COS:
            result = cosf(popExLinear(exStack));
            break;
        #endif

        #ifdef TAN
        case TAN:
            result = tanf(popExLinear(exStack));
            break;
        #endif

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                // Leitura dos canais R (x) e G (y) contidos em cada pixel
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        /*
        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = tanhf(sum);
            break;
        #endif
        */

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearActiveImageQuarter(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;

    // Mapeamento do ID do bloco no CUDA (equivalente ao get_group_id(0))
    int group_id = blockIdx.x;

    // Leitura do tipo da função no primeiro pixel (0, node+1, group_id)
    uint4 pixelInt = tex3D<uint4>(c, 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
    float4 pixelFloat;

    unsigned int inputs = MAX_ARITY; // c->nodes[node].maxInputs;

    switch (pixelInt.x) {

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                // Leitura de todos os 4 canais RGBA (.x, .y, .z, .w) em um único acesso de textura
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        /*
        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 1) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            result = tanhf(sum);
            break;
        #endif
        */

        default:
            break;
    }
    return result;
}

static __device__ float executeFunctionLinearActiveImageCompact(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    int group_id = blockIdx.x;

    uint4 pixelInt;
    float4 pixelFloat;

    // Leitura no deslocamento x=1 para a versão "Compact"
    pixelInt = tex3D<uint4>(c, 1.0f + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);

    unsigned int inputs = MAX_ARITY; // c->nodes[node].maxInputs;
    switch (pixelInt.x) {

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 2) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = 1.0f / (1.0f + expf(-sum));
            break;
        #endif

        #ifdef GAUSS
        case GAUSS:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 2) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = expf(-(powf(sum - 0.0f, 2.0f)) / (2.0f * powf(1.0f, 2.0f)));
            break;
        #endif

        #ifdef STEP
        case STEP:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 2) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            if (sum < 0.0f) {
                result = 0.0f;
            } else {
                result = 1.0f;
            }
            break;
        #endif

        #ifdef SOFTSIGN
        case SOFTSIGN:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 2) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = sum / (1.0f + fabsf(sum));
            break;
        #endif

        #ifdef TANH
        case TANH:
            sum = 0.0f;
            for (i = 0; i < inputs; i++) {
                pixelFloat = tex3D<float4>(c, (float)(i + 2) + 0.5f, (float)(node + 1) + 0.5f, (float)group_id + 0.5f);
                sum += (popExLinear(exStack) * pixelFloat.x);
            }
            result = tanhf(sum);
            break;
        #endif

        default:
            break;
    }
    return result;
}

// ============================================================================
// KERNELS DE AVALIAÇÃO (CUDA KERNELS)
// ============================================================================

/*PADRÃO*/

__device__ void evaluateCircuit(Chromosome* c,
                                float* data,
                                float* out,
                                float* fitness) {

    // Alocação da memória compartilhada por bloco (equivalente ao __local de OpenCL)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    /* Quando NUM_POINTS é divisível por LOCAL_SIZE, evitamos a verificação no loop */
    for (k = 0; k < (M / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[currentActive].inputs[j] >= N) { // nó intermediário
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                if (out[k * LOCAL_SIZE + local_id + (M * i)] == alreadyEvaluated[c->output[i]]) {
                    erro += 1;
                }
            }

    #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela da memória compartilhada por Bloco
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0];
    }
}


__device__ void evaluateCircuitTrainValidation(Chromosome* c,
                                                float* data,
                                                float* out,
                                                float* fitness) {

    // Alocação da memória compartilhada por bloco (equivalente ao __local de OpenCL)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[currentActive].inputs[j] >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela da memória compartilhada por Bloco
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}


__device__ void evaluateCircuitTest(Chromosome* c,
                                    float* data,
                                    float* out,
                                    float* fitness) {

    // Alocação da memória compartilhada (equivalente ao __local no OpenCL)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TEST / LOCAL_SIZE_TEST); k++) {

    #else
        for (k = 0; k < ceilf(M_TEST / (float)LOCAL_SIZE_TEST); k++) {

            if (k * LOCAL_SIZE_TEST + local_id < M_TEST) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[currentActive].inputs[j] >= N) { // nó intermediário
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TEST + local_id + (M_TEST * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_TEST + local_id + (M_TEST * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela na memória compartilhada
    for (i = LOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TEST_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TEST))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_TEST;
    }
}

__device__ void evaluateCircuitTrain(Chromosome* c,
                                     float* data,
                                     float* out,
                                     float* fitness) {

    // Alocação da memória compartilhada por bloco (substitui o __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TRAIN / LOCAL_SIZE_TRAIN); k++) {

    #else
        for (k = 0; k < ceilf(M_TRAIN / (float)LOCAL_SIZE_TRAIN); k++) {

            if (k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[currentActive].inputs[j] >= N) { // nó intermediário
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * i)] == alreadyEvaluated[c->output[i]]) {
                    erro += 1;
                }
            }

    #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela na memória compartilhada por bloco
    for (i = LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN))
    #endif
            error[local_id] += error[local_id + i];
    }

    if (local_id == 0) {
        fitness[group_id] = error[0];
    }
}

__device__ void evaluateCircuitValidation(Chromosome* c,
                                          float* data,
                                          float* out,
                                          float* fitness) {

    // Alocação da memória compartilhada por bloco (substitui o __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->activeNodes[i];
                activeInputs = c->nodes[currentActive].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[currentActive].inputs[j] >= N) { // nó intermediário
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[currentActive].inputs[j] - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * c->nodes[currentActive].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinear(c, currentActive, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela na memória compartilhada por bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*PADRÃO*/

/*==============================================================================================================================================================================================================*/

/*ATIVADO*/

__device__ void evaluateCircuitActive(ActiveChromosome* c,
                                     float* data,
                                     float* out,
                                     float* fitness) {

    // Em CUDA, a memória compartilhada por bloco pode ser declarada assim:
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;
    int local_size = blockDim.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    /* Quando NUM_POINTS é divisível por blockDim.x */
    for(k = 0; k < (M / local_size); k++) {
    #else
    for(k = 0; k < (int)ceilf(M / (float)local_size); k++) {
        if(k * local_size + local_id < M) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for(i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++) {
                    if (c->nodes[i].inputs[j] >= N) { // se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[i].inputs[j] - N]);
                    } else {
                        pushExLinear(&exStack, data[k * local_size + local_id + (M * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);
            }

            for(i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if(alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * local_size + local_id + (M * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1.0f : 0.0f;

    #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução de erros por bloco (Work Group)
    for(i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if(local_id < i)
    #else
        if((local_id < i) && (local_id + i < local_size))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if(local_id == 0) {
        fitness[group_id] = error[0] / (float)M;
    }
}

__device__ void evaluateCircuitTrainValidationActive(ActiveChromosome* c,
                                                      float* data,
                                                      float* out,
                                                      float* fitnessValidation) {

    // Memória compartilhada por bloco (substitui o __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < c->numActiveNodes; i++) {
                currentActive = c->nodes[i].originalIndex;
                activeInputs = c->nodes[i].maxInputs;

                for (j = 0; j < activeInputs; j++) {
                    if (c->nodes[i].inputs[j] >= N) { // se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[c->nodes[i].inputs[j] - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M_VALIDATION * c->nodes[i].inputs[j])]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActive(c, i, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela na memória compartilhada
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitnessValidation[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*==============================================================================================================================================================================================================*/

/*COMPACTO*/


__device__ void evaluateCircuitTrainCompact(CompactChromosome* c,
                                             float* data,
                                             float* out,
                                             float* fitness) {

    // Memória compartilhada por bloco (substitui o __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TRAIN / LOCAL_SIZE_TRAIN); k++) {

    #else
        for (k = 0; k < ceilf(M_TRAIN / (float)LOCAL_SIZE_TRAIN); k++) {

            if (k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < (int)ceilf(c->numActiveNodes / 2.0f); i++) {

                currentActive0 = c->activeNodes[i] >> 16;
                currentActive1 = c->activeNodes[i] & 0xFFFF;
                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    input0 = (c->nodes[currentActive0].inputs[j] >> 16);
                    input1 = (c->nodes[currentActive0].inputs[j] & 0xFFFF);

                    if (input0 >= N) { // se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearCompact(c, currentActive0, &exStack);

                if (2 * (i + 1) > c->numActiveNodes) break;

                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    input0 = (c->nodes[currentActive1].inputs[j] >> 16);
                    input1 = (c->nodes[currentActive1].inputs[j] & 0xFFFF);

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearCompact(c, currentActive1, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela
    for (i = LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN))
    #endif
            error[local_id] += error[local_id + i];
    }

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_TRAIN;
    }
}


__device__ void evaluateCircuitValidationCompact(CompactChromosome* c,
                                                  float* data,
                                                  float* out,
                                                  float* fitness) {

    // Memória compartilhada por bloco em CUDA (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            for (i = 0; i < (int)ceilf(c->numActiveNodes / 2.0f); i++) {

                currentActive0 = c->activeNodes[i] >> 16;
                currentActive1 = c->activeNodes[i] & 0xFFFF;

                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    input0 = (c->nodes[currentActive0].inputs[j] >> 16);
                    input1 = (c->nodes[currentActive0].inputs[j] & 0xFFFF);

                    if (input0 >= N) { // se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else { // entrada da rede/dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearCompact(c, currentActive0, &exStack);

                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    input0 = (c->nodes[currentActive1].inputs[j] >> 16);
                    input1 = (c->nodes[currentActive1].inputs[j] & 0xFFFF);

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearCompact(c, currentActive1, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                unsigned int nodeIndex = c->output[i];

                if (alreadyEvaluated[nodeIndex] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[nodeIndex];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*COMPACTO*/

/*==============================================================================================================================================================================================================*/

/*IMAGEM R*/


__device__ void evaluateCircuitImage_R(cudaTextureObject_t indiv,
                                        float* data,
                                        float* out,
                                        float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            // tex2DLayered<uint4> substitui o read_imageui para arrays 2D de imagens
            // Sintaxe: tex2DLayered<uint4><T>(texObj, x, y, layer)
            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {
                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    if (pixel.x >= N) { // Se for outro nó, pega o resultado calculado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else { // Entrada de dados da rede
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(indiv, indexCalc, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                pixel = tex2DLayered<uint4>(indiv, i + 1, 0, group_id);

                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE + local_id + (M * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M;
    }
}

__device__ void evaluateCircuitValidationImage_R(cudaTextureObject_t indiv,
                                                 float* data,
                                                 float* out,
                                                 float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            // Substitui read_imageui por tex2DLayered<uint4>(texObj, x, y, layer)
            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {
                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    if (pixel.x >= N) { // Se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else { // Entrada de dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.x)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImage(indiv, indexCalc, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                pixel = tex2DLayered<uint4>(indiv, i + 1, 0, group_id);

                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*IMAGEM R*/

/*==============================================================================================================================================================================================================*/

/*IMAGEM RG*/


__device__ void evaluateCircuitImage_RG(cudaTextureObject_t indiv,
                                         float* data,
                                         float* out,
                                         float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            uint4 pixel;
            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {

                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY / 2;

                for (j = 0; j < activeInputs; j++) {

                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    // Canal X
                    if (pixel.x >= N) { // Se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else { // Entrada de dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.x)]);
                    }

                    // Canal Y
                    if (pixel.y >= N) { // Se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                    } else { // Entrada de dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.y)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageHalf(indiv, indexCalc, &exStack);
            }

            int index = 1;
            for (i = 0; i < MAX_OUTPUTS; i += 2) {
                pixel = tex2DLayered<uint4>(indiv, index, 0, group_id);
                index++;

                // Saída correspondente ao canal X
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE + local_id + (M * i)] == 1.0f) ? i : correctClass;

                if (i + 1 >= MAX_OUTPUTS) break;

                // Saída correspondente ao canal Y
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }

                correctClass = (out[k * LOCAL_SIZE + local_id + (M * (i + 1))] == 1.0f) ? (i + 1) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M;
    }
}


__device__ void evaluateCircuitValidationImage_RG(cudaTextureObject_t indiv,
                                                   float* data,
                                                   float* out,
                                                   float* fitness) {

    // Memória compartilhada dinamicamente/estaticamente alocada por bloco
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {
                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY / 2;

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    // Leitura do canal X
                    if (pixel.x >= N) { // Se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else { // Entrada de dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.x)]);
                    }

                    // Leitura do canal Y
                    if (pixel.y >= N) { // Se é outro nó, empilha resultado
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                    } else { // Entrada de dados
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.y)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageHalf(indiv, indexCalc, &exStack);
            }

            int index = 1;
            for (i = 0; i < MAX_OUTPUTS; i += 2) {
                pixel = tex2DLayered<uint4>(indiv, index, 0, group_id);
                index++;

                // Saída para o canal X
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;

                if (i + 1 == MAX_OUTPUTS) break;

                // Saída para o canal Y
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 1))] == 1.0f) ? (i + 1) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*IMAGEM RG*/

/*==============================================================================================================================================================================================================*/

/*IMAGEM RGBA*/

__device__ void evaluateCircuitImage_RGBA(cudaTextureObject_t indiv,
                                          float* data,
                                          float* out,
                                          float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M / LOCAL_SIZE); k++) {

    #else
        for (k = 0; k < ceilf(M / (float)LOCAL_SIZE); k++) {

            if (k * LOCAL_SIZE + local_id < M) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;

            uint4 pixel;
            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {

                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY / 4;

                for (j = 0; j < activeInputs; j++) {

                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    // Canal X
                    if (pixel.x >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.x)]);
                    }

                    // Canal Y
                    if (pixel.y >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.y)]);
                    }

                    // Canal Z
                    if (pixel.z >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.z - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.z)]);
                    }

                    // Canal W
                    if (pixel.w >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.w - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE + local_id + (M * pixel.w)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageQuarter(indiv, indexCalc, &exStack);
            }

            int index = 1;
            for (i = 0; i < MAX_OUTPUTS; i += 4) {
                pixel = tex2DLayered<uint4>(indiv, index, 0, group_id);
                index++;

                // Canal X
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }
                correctClass = (out[k * LOCAL_SIZE + local_id + (M * i)] == 1.0f) ? i : correctClass;

                if (i + 1 >= MAX_OUTPUTS) break;

                // Canal Y
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }
                correctClass = (out[k * LOCAL_SIZE + local_id + (M * (i + 1))] == 1.0f) ? (i + 1) : correctClass;

                if (i + 2 >= MAX_OUTPUTS) break;

                // Canal Z
                if (alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i + 2;
                }
                correctClass = (out[k * LOCAL_SIZE + local_id + (M * (i + 2))] == 1.0f) ? (i + 2) : correctClass;

                if (i + 3 >= MAX_OUTPUTS) break;

                // Canal W
                if (alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i + 3;
                }
                correctClass = (out[k * LOCAL_SIZE + local_id + (M * (i + 3))] == 1.0f) ? (i + 3) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M;
    }
}


__device__ void evaluateCircuitValidationImage_RGBA(cudaTextureObject_t indiv,
                                                     float* data,
                                                     float* out,
                                                     float* fitness) {

    // Memória compartilhada dinamicamente alocada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    int currentActive, activeInputs;
    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;

    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif
            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < activenodes; i++) {
                int indexCalc = (2 * i) + 1;
                pixel = tex2DLayered<uint4>(indiv, 0, indexCalc, group_id);

                currentActive = pixel.x;
                activeInputs = MAX_ARITY / 4;

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 1, indexCalc, group_id);

                    // Canal X
                    if (pixel.x >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.x - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.x)]);
                    }

                    // Canal Y
                    if (pixel.y >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.y - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.y)]);
                    }

                    // Canal Z
                    if (pixel.z >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.z - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.z)]);
                    }

                    // Canal W
                    if (pixel.w >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[pixel.w - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * pixel.w)]);
                    }
                }

                alreadyEvaluated[currentActive] = executeFunctionLinearActiveImageQuarter(indiv, indexCalc, &exStack);
            }

            int index = 1;
            for (i = 0; i < MAX_OUTPUTS; i += 2) {
                pixel = tex2DLayered<uint4>(indiv, index, 0, group_id);
                index++;

                // Canal X
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }
                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;

                if (i + 1 == MAX_OUTPUTS) break;

                // Canal Y
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }
                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 1))] == 1.0f) ? (i + 1) : correctClass;

                if (i + 2 == MAX_OUTPUTS) break;

                // Canal Z
                if (alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i + 2;
                }
                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 2))] == 1.0f) ? (i + 2) : correctClass;

                if (i + 3 == MAX_OUTPUTS) break;

                // Canal W
                if (alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i + 3;
                }
                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 3))] == 1.0f) ? (i + 3) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*IMAGEM RGBA*/

/*==============================================================================================================================================================================================================*/

/*COMPACTA IMAGE_R */


__device__ void evaluateCircuitTrainCompactImage_R(cudaTextureObject_t indiv,
                                                    float* data,
                                                    float* out,
                                                    float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TRAIN / LOCAL_SIZE_TRAIN); k++) {

    #else
        for (k = 0; k < ceilf(M_TRAIN / (float)LOCAL_SIZE_TRAIN); k++) {

            if (k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < ceilf(activenodes / (float)2); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);

                if (2 * (i + 1) > activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                pixel = tex2DLayered<uint4>(indiv, i + 1, 0, group_id);

                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_TRAIN;
    }
}


__device__ void evaluateCircuitValidationCompactImage_R(cudaTextureObject_t indiv,
                                                         float* data,
                                                         float* out,
                                                         float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < ceilf(activenodes / (float)2); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);

                if (2 * (i + 1) > activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = MAX_ARITY;

                for (j = 0; j < activeInputs / 2; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageCompact(indiv, indexCalc, &exStack);
            }

            for (i = 0; i < MAX_OUTPUTS; i++) {
                pixel = tex2DLayered<uint4>(indiv, i + 1, 0, group_id);

                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*COMPACTA IMAGE_R */

/*==============================================================================================================================================================================================================*/

/*COMPACTA IMAGE_RG */



static __device__ float executeFunctionLinearActiveImageHalfCompact(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum;
    int group_id = blockIdx.x;

    uint4 pixelInt;
    float4 pixelFloat;

    // Leitura dos dados inteiros da textura (ex: código do operador)
    pixelInt = tex2DLayered<uint4>(c, 1, node + 1, group_id);

    unsigned int inputs = MAX_ARITY;

    switch (pixelInt.x) {

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs / 2; i++) {
                // Leitura dos dados de ponto flutuante (pesos/parâmetros) da textura
                pixelFloat = tex2DLayered<float4>(c, i + 2, node + 1, group_id);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
            }
            result = 1.0f / (1.0f + __expf(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}


__device__ void evaluateCircuitTrainCompactImage_RG(cudaTextureObject_t indiv,
                                                     float* data,
                                                     float* out,
                                                     float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TRAIN / LOCAL_SIZE_TRAIN); k++) {

    #else
        for (k = 0; k < ceilf(M_TRAIN / (float)LOCAL_SIZE_TRAIN); k++) {

            if (k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < ceilf(activenodes / (float)4); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                // --- Nó 1 do pacote ---
                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Processa canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Processa canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Nó 2 do pacote ---
                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Processa canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Processa canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Nó 3 do pacote ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Processa canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Processa canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;

                if (cont == activenodes) break;

                // --- Nó 4 do pacote ---
                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Processa canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Processa canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
            }

            // --- Leitura das Saídas (Canais X e Y de cada pixel) ---
            int index = 0;
            for (i = 0; i < MAX_OUTPUTS; i += 2) {
                pixel = tex2DLayered<uint4>(indiv, index + 1, 0, group_id);
                index++;

                // Saída par (canal X)
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * i)] == 1.0f) ? i : correctClass;

                if (i + 1 >= MAX_OUTPUTS) break;

                // Saída ímpar (canal Y)
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * (i + 1))] == 1.0f) ? (i + 1) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela por Bloco
    for (i = LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_TRAIN;
    }
}


__device__ void evaluateCircuitValidationCompactImage_RG(cudaTextureObject_t indiv,
                                                          float* data,
                                                          float* out,
                                                          float* fitness) {

    // Memória compartilhada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < ceilf(activenodes / (float)4); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                // --- Nó 1 do pacote ---
                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Nó 2 do pacote ---
                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    // Canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Nó 3 do pacote ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    // Canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;

                if (cont == activenodes) break;

                // --- Nó 4 do pacote ---
                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)4);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    // Canal Y
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageHalfCompact(indiv, indexCalc, &exStack);
                cont++;
            }

            // --- Leitura das Saídas ---
            int index = 0;
            for (i = 0; i < MAX_OUTPUTS; i += 2) {
                pixel = tex2DLayered<uint4>(indiv, index + 1, 0, group_id);
                index++;

                // Saída par
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;

                if (i + 1 >= MAX_OUTPUTS) break;

                // Saída ímpar
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 1))] == 1.0f) ? (i + 1) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1 : 0;

    #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
    #endif
    }

    error[local_id] = (float)erro;
    __syncthreads();

    // Redução Paralela
    for (i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_VALIDATION;
    }
}

/*COMPACTA IMAGE_RG */

/*==============================================================================================================================================================================================================*/

/*COMPACTA IMAGE_RGBA */


static __device__ float executeFunctionLinearActiveImageQuarterCompact(cudaTextureObject_t c, int node, ExStackLinear* exStack) {
    int i;
    float result = 0.0f, sum = 0.0f;
    int group_id = blockIdx.x;

    uint4 pixelInt;
    float4 pixelFloat;

    // Leitura do tipo unsigned int (imagem 2D Array / Layered)
    pixelInt = tex2DLayered<uint4>(c, 1, node + 1, group_id);

    unsigned int inputs = MAX_ARITY;

    switch (pixelInt.x) {

        #ifdef ONE
        case ONE:
            result = 1.0f;
            break;
        #endif

        #ifdef ZERO
        case ZERO:
            result = 0.0f;
            break;
        #endif

        #ifdef PI
        case PI:
            result = CONST_PI;
            break;
        #endif

        #ifdef WIRE
        case WIRE:
            result = popExLinear(exStack);
            break;
        #endif

        #ifdef SIG
        case SIG:
            sum = 0.0f;
            for (i = 0; i < inputs / 4; i++) {
                // Leitura no formato de ponto flutuante (float4)
                pixelFloat = tex2DLayered<float4>(c, i + 2, node + 1, group_id);
                sum += (popExLinear(exStack) * pixelFloat.x);
                sum += (popExLinear(exStack) * pixelFloat.y);
                sum += (popExLinear(exStack) * pixelFloat.z);
                sum += (popExLinear(exStack) * pixelFloat.w);
            }
            result = 1.0f / (1.0f + __expf(-sum));
            break;
        #endif

        default:
            break;
    }
    return result;
}

__device__ void evaluateCircuitTrainCompactImage_RGBA(cudaTextureObject_t indiv,
                                                      float* data,
                                                      float* out,
                                                      float* fitness) {

    // Memória compartilhada dinamicamente alocada por bloco (substitui __local float* error)
    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    for (k = 0; k < (M_TRAIN / LOCAL_SIZE_TRAIN); k++) {

    #else
        for (k = 0; k < ceilf(M_TRAIN / (float)LOCAL_SIZE_TRAIN); k++) {

            if (k * LOCAL_SIZE_TRAIN + local_id < M_TRAIN) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            // Leitura inicial de nós ativos
            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            // Loop de processamento de pacotes de 8 nós
            for (i = 0; i < ceilf(activenodes / (float)8); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                // --- Nó 1 do pacote (canal X - parte alta e baixa) ---
                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 2 e transiciona para o canal Y (Nó 3)
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 3 (canal Y - parte alta) ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
// --- Finalização do Nó 1 e processamento do Nó 2 (canal X.2) ---
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 2 e transiciona para o canal Y (Nó 3)
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 3 (canal Y - parte alta) ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 2 e transiciona para o canal Y (Nó 3)
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 3 (canal Y - parte alta) ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

            alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 2 e transiciona para o canal Y (Nó 3)
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 3 (canal Y - parte alta) ---
                currentActive0 = pixelAux.y >> 16;
                currentActive1 = pixelAux.y & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 4 (canal Y.2) e transiciona para o canal Z
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 5 (canal Z - parte alta) ---
                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                // Finaliza avaliação do Nó 5 (canal Z.2) e transiciona para o canal W
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 6 (canal W - parte alta) ---
                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & 0xFFFF;

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }

                // Finaliza avaliação do Nó 6 (canal W.1) e transiciona para o Nó 7 (canal W.2)
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if (cont == activenodes) break;

                // --- Início do Nó 7 (canal W - parte baixa) ---
                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);
                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    // Canal X (input0 e input1)
                    input0 = pixel.x >> 16;
                    input1 = pixel.x & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal Y (input0 e input1)
                    input0 = pixel.y >> 16;
                    input1 = pixel.y & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    // Canal Z (input0 e input1)
                    input0 = pixel.z >> 16;
                    input1 = pixel.z & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }

                    // Canal W (input0 e input1)
                    input0 = pixel.w >> 16;
                    input1 = pixel.w & 0xFFFF;

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * input1)]);
                    }
                }
                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);

            } // Fim do loop principal de nós ativos

            int index = 0;
            for (i = 0; i < MAX_OUTPUTS; i += 4) {
                // Leitura de textura layered no CUDA
                pixel = tex2DLayered<uint4>(indiv, index + 1, 0, group_id);
                index++;

                // Canal X
                if (alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * i)] == 1.0f) ? i : correctClass;

                if (i + 1 >= MAX_OUTPUTS) break;

                // Canal Y
                if (alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * (i + 1))] == 1.0f) ? (i + 1) : correctClass;

                if (i + 2 >= MAX_OUTPUTS) break;

                // Canal Z
                if (alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i + 2;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * (i + 2))] == 1.0f) ? (i + 2) : correctClass;

                if (i + 3 >= MAX_OUTPUTS) break;

                // Canal W
                if (alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i + 3;
                }

                correctClass = (out[k * LOCAL_SIZE_TRAIN + local_id + (M_TRAIN * (i + 3))] == 1.0f) ? (i + 3) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1.0f : 0.0f;

        #ifdef NUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
        #endif
    }

    // Armazena no array de memória compartilhada (shared memory)
    error[local_id] = erro;
    __syncthreads();

    // Redução paralela dos erros por bloco (Work-Group)
    for (i = LOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2
        if (local_id < i)
    #else
        /* LOCAL_SIZE não é potência de 2: checagem para evitar acesso fora dos limites */
        if ((local_id < i) && (local_id + i < LOCAL_SIZE_TRAIN))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    // O thread 0 do bloco escreve o resultado global do fitness
    if (local_id == 0) {
        fitness[group_id] = error[0] / (float)M_TRAIN;
    }
}

__device__ void evaluateCircuitValidationCompactImage_RGBA(cudaTextureObject_t indiv,
                                                           const float* __restrict__ data,
                                                           const float* __restrict__ out,
                                                           float* fitness) {

    extern __shared__ float error[];

    int i, k, j = 0;
    unsigned int currentActive0, currentActive1, activeInputs;
    unsigned int input0, input1;

    int erro = 0;

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;

    error[local_id] = 0.0f;

    float maxPredicted;
    int predictedClass;
    int correctClass;
    int indexCalc;
    float alreadyEvaluated[MAX_NODES];

    #ifndef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
    /* Quando sabemos que NUM_POINTS é divisível por LOCAL_SIZE, evitamos a
       comparação em cada iteração. */
    for (k = 0; k < (M_VALIDATION / LOCAL_SIZE_VALIDATION); k++) {

    #else
        for (k = 0; k < ceilf(M_VALIDATION / (float)LOCAL_SIZE_VALIDATION); k++) {

            if (k * LOCAL_SIZE_VALIDATION + local_id < M_VALIDATION) {
    #endif

            maxPredicted = -FLT_MAX;
            predictedClass = 0;
            correctClass = 0;

            ExStackLinear exStack;
            exStack.topIndex = -1;
            uint4 pixel;
            int cont = 0;

            pixel = tex2DLayered<uint4>(indiv, 0, 0, group_id);
            int activenodes = pixel.x;

            for (i = 0; i < ceilf(activenodes / (float)8); i++) {
                uint4 pixelAux;
                pixelAux = tex2DLayered<uint4>(indiv, 0, i + 1, group_id);

                currentActive0 = pixelAux.x >> 16;
                currentActive1 = pixelAux.x & (0xFFFF);

                indexCalc = (2 * currentActive0) + 1;

                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = (pixel.x >> 16);
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é outro nó, empilha resultado prévio
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16);
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16);
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16);
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if (cont == activenodes) break;

                indexCalc = (2 * currentActive1) + 1;
                pixel = tex2DLayered<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceilf(MAX_ARITY / (float)8);

                for (j = 0; j < activeInputs; j++) {
                    pixel = tex2DLayered<uint4>(indiv, j + 2, indexCalc, group_id);

                    input0 = (pixel.x >> 16);
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16);
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    if (j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16);
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16);
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) {
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }
alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = tex3D<uint4>(indiv, 1, indexCalc, group_id);

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = tex3D<uint4>(indiv, j+2, indexCalc, group_id);

                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.z >> 16;
                currentActive1 = pixelAux.z & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = tex3D<uint4>(indiv, 1, indexCalc, group_id);
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = tex3D<uint4>(indiv, j+2, indexCalc, group_id);

                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont  == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = tex3D<uint4>(indiv, 1, indexCalc, group_id);

               activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = tex3D<uint4>(indiv, j+2, indexCalc, group_id);

                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;
                if(cont  == activenodes) break;

                currentActive0 = pixelAux.w >> 16;
                currentActive1 = pixelAux.w & (0xFFFF);
                /*if(group_id == 0 && local_id == 0)
                    printf("%d %d \n",currentActive0, currentActive1);
                */
                indexCalc = (2*currentActive0)+1;

                pixel = tex3D<uint4>(indiv, 1, indexCalc, group_id);
                /*
                if(group_id == 0 && local_id == 0)
                    printf("%d\n", pixel.x);
                */
                activeInputs = ceil(MAX_ARITY/(float)8);// c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = tex3D<uint4>(indiv, j+2, indexCalc, group_id);

                    input0 = (pixel.x >> 16 );
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16 );
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16 );
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16 );
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);

                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + ( M_VALIDATION * input1)]);
                    }
                }
                alreadyEvaluated[currentActive0] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
                cont++;

                if(cont == activenodes) break;

                indexCalc = (2*currentActive1)+1;
                pixel = tex3D<uint4>(indiv, 1, indexCalc, group_id);

                activeInputs = ceil(MAX_ARITY/(float)8); // c->nodes[i].maxInputs;

                for(j = 0; j < activeInputs; j++){
                    pixel = tex3D<uint4>(indiv, j+2, indexCalc, group_id);

                    input0 = (pixel.x >> 16);
                    input1 = (pixel.x & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.y >> 16);
                    input1 = (pixel.y & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    if(j == activeInputs - 1) break;

                    input0 = (pixel.z >> 16);
                    input1 = (pixel.z & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }

                    input0 = (pixel.w >> 16);
                    input1 = (pixel.w & (0xFFFF));

                    if (input0 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input0 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input0)]);
                    }

                    if (input1 >= N) { // se é um outro nó, empilha nó ou o resultado
                        pushExLinear(&exStack, alreadyEvaluated[input1 - N]);
                    } else {
                        pushExLinear(&exStack, data[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * input1)]);
                    }
                }

                alreadyEvaluated[currentActive1] = executeFunctionLinearActiveImageQuarterCompact(indiv, indexCalc, &exStack);
            }

            int index = 0;
            for(i = 0; i < MAX_OUTPUTS; i += 4) {
                pixel = tex3D<uint4>(indiv, index+1, 0, group_id);
                index++;

                if(alreadyEvaluated[pixel.x] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.x];
                    predictedClass = i;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * i)] == 1.0f) ? i : correctClass;

                if(i + 1 >= MAX_OUTPUTS) break;

                if(alreadyEvaluated[pixel.y] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.y];
                    predictedClass = i + 1;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 1))] == 1.0f) ? (i + 1) : correctClass;

                if(i + 2 >= MAX_OUTPUTS) break;

                if(alreadyEvaluated[pixel.z] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.z];
                    predictedClass = i + 2;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 2))] == 1.0f) ? (i + 2) : correctClass;

                if(i + 3 >= MAX_OUTPUTS) break;

                if(alreadyEvaluated[pixel.w] > maxPredicted) {
                    maxPredicted = alreadyEvaluated[pixel.w];
                    predictedClass = i + 3;
                } else {
                    maxPredicted = maxPredicted;
                    predictedClass = predictedClass;
                }

                correctClass = (out[k * LOCAL_SIZE_VALIDATION + local_id + (M_VALIDATION * (i + 3))] == 1.0f) ? (i + 3) : correctClass;
            }

            erro += (predictedClass == correctClass) ? 1.0f : 0.0f;

        #ifdef NUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
        }
        #endif
    }

    error[local_id] = erro;
    __syncthreads();

    /// redução de erros por bloco de threads (work group)
    for(i = LOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2 / 2; i > 0; i >>= 1) {
        __syncthreads();

    #ifndef LOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2
        if(local_id < i)
    #else
        /* LOCAL_SIZE não é potência de 2, validação extra de limites */
        if((local_id < i) && (local_id + i < LOCAL_SIZE_VALIDATION))
    #endif
            error[local_id] += error[local_id + i];
    }

    __syncthreads();

    if(local_id == 0) {
        fitness[group_id] = error[0] / M_VALIDATION;
    }
}

/*COMPACTA IMAGE_RGBA */

/*==============================================================================================================================================================================================================*/

/*PADRÂO*/


__global__ void evolve(unsigned int* functionSet,
                        int *seeds,
                        Chromosome* population,
                        Chromosome* best) {

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("%d\n", local_id);
    int seed = seeds[global_id];
    // int seed = seeds[group_id];
    // printf("%d\n", local_id);
    // if(local_id == 0)
    //     population[group_id] = *best;

    // __syncthreads();
    // mutateTopologyProbabilistic(&newBest[group_id], functionSet, &seed, 0);
    mutateTopologyProbabilistic2(&population[group_id], functionSet, &seed, 0);

    // __syncthreads();

    // if(local_id == 0)
    seeds[global_id] = seed;
}

__global__ void evaluateTest(float* dataset,
                             float* outputs,
                             unsigned int* functionSet,
                             Chromosome* individual,
                             float* fitness,
                             float* fitnessValidation) {

    // Alocação de memória compartilhada (__local em OpenCL)
    extern __shared__ float error[];

    evaluateCircuitTest(individual, dataset, outputs, fitness);
}

__global__ void evaluateTrain(float* dataset,
                              float* outputs,
                              unsigned int* functionSet,
                              Chromosome* pop,
                              float* fitness,
                              float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    evaluateCircuit(&pop[group_id], dataset, outputs, fitness);
}

__global__ void evaluateValidation(float* dataset,
                                   float* outputs,
                                   unsigned int* functionSet,
                                   Chromosome* pop,
                                   float* fitness,
                                   float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    evaluateCircuitValidation(&pop[group_id], dataset, outputs, fitnessValidation);
}

__global__ void evaluateTrainValidation(float* datasetTrain,
                                        float* outputsTrain,
                                        float* datasetValid,
                                        float* outputsValid,
                                        unsigned int* functionSet,
                                        Chromosome* pop,
                                        Chromosome* best,
                                        float* fitness,
                                        float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    // pop[group_id] = *best;
    // __syncthreads();

    evaluateCircuit(&pop[group_id], datasetTrain, outputsTrain, fitness);

    __syncthreads();

    evaluateCircuitTrainValidation(&pop[group_id], datasetValid, outputsValid, fitnessValidation);
}

/*PADRÂO*/

/*==============================================================================================================================================================================================================*/

/*COMPACTO*/


__global__ void evaluateTrainCompact(float* dataset,
                                     float* outputs,
                                     unsigned int* functionSet,
                                     CompactChromosome* pop,
                                     float* fitness,
                                     float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;

    evaluateCircuitTrainCompact(&pop[group_id], dataset, outputs, fitness);
}

__global__ void evaluateValidationCompact(float* dataset,
                                          float* outputs,
                                          unsigned int* functionSet,
                                          CompactChromosome* pop,
                                          float* fitness,
                                          float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;

    evaluateCircuitValidationCompact(&pop[group_id], dataset, outputs, fitnessValidation);
}

/*COMPACTO*/

/*==============================================================================================================================================================================================================*/

/*ATIVO*/

__global__ void evaluateTrainValidationActive(float* datasetTrain,
                                              float* outputsTrain,
                                              float* datasetValid,
                                              float* outputsValid,
                                              unsigned int* functionSet,
                                              ActiveChromosome* pop,
                                              float* fitness,
                                              float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    // pop[group_id] = *best;
    // __syncthreads();

    evaluateCircuitActive(&pop[group_id], datasetTrain, outputsTrain, fitness);

    __syncthreads();

    evaluateCircuitTrainValidationActive(&pop[group_id], datasetValid, outputsValid, fitnessValidation);
}

/*ATIVO*/

/*==============================================================================================================================================================================================================*/

/*IMAGE_R*/


__global__ void evaluateTrainImage(float* data,
                                   float* out,
                                   unsigned int* functionSet,
                                   cudaTextureObject_t pop,
                                   float* fitness,
                                   float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 1, 1, group_id);
    if(local_id == 0)
        printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
    */
    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitImage_R(pop, data, out, fitness);
}

__global__ void evaluateValidationImage(float* data,
                                        float* out,
                                        unsigned int* functionSet,
                                        cudaTextureObject_t pop,
                                        float* fitness,
                                        float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationImage_R(pop, data, out, fitnessValidation);
}

/*IMAGE_R*/

/*==============================================================================================================================================================================================================*/

/*IMAGE_RG*/


__global__ void evaluateTrainImageHalf(float* data,
                                       float* out,
                                       unsigned int* functionSet,
                                       cudaTextureObject_t pop,
                                       float* fitness,
                                       float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 0, 0, group_id);
    if(local_id == 0 ){
        int i;
        for(i = 0; i < 11; i ++){
            pixel = tex3D(pop, i, 603, group_id);
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        }
    }
    */

    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitImage_RG(pop, data, out, fitness);
}

__global__ void evaluateValidationImageHalf(float* data,
                                            float* out,
                                            unsigned int* functionSet,
                                            cudaTextureObject_t pop,
                                            float* fitness,
                                            float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationImage_RG(pop, data, out, fitnessValidation);
}

/*IMAGE_RG*/
/*==============================================================================================================================================================================================================*/

/*IMAGE_RGBA*/


__global__ void evaluateTrainImageQuarter(float* data,
                                          float* out,
                                          unsigned int* functionSet,
                                          cudaTextureObject_t pop,
                                          float* fitness,
                                          float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 0, 0, group_id);
    if(local_id == 0 ){
        int i;
        for(i = 0; i < 11; i ++){
            pixel = tex3D(pop, i, 603, group_id);
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        }
    }
    */

    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitImage_RGBA(pop, data, out, fitness);
}

__global__ void evaluateValidationImageQuarter(float* data,
                                               float* out,
                                               unsigned int* functionSet,
                                               cudaTextureObject_t pop,
                                               float* fitness,
                                               float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationImage_RGBA(pop, data, out, fitnessValidation);
}

/*IMAGE_RGBA*/

/*==============================================================================================================================================================================================================*/

/*COMPACTA IMAGE_R*/

__global__ void evaluateTrainImageCompact(float* data,
                                          float* out,
                                          unsigned int* functionSet,
                                          cudaTextureObject_t pop,
                                          float* fitness,
                                          float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 0, 0, group_id);
    if(local_id == 0 ){
        // printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        // for(i = 0; i < 11; i ++){
            pixel = tex3D(pop, 0, 1, group_id);
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        // }
    }
    */

    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitTrainCompactImage_R(pop, data, out, fitness);
}

__global__ void evaluateValidationImageCompact(float* data,
                                               float* out,
                                               unsigned int* functionSet,
                                               cudaTextureObject_t pop,
                                               float* fitness,
                                               float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationCompactImage_R(pop, data, out, fitnessValidation);
}

/*COMPACTA IMAGE_R*/

/*==============================================================================================================================================================================================================*/

/**COMPACTA IMAGE_RG */


__global__ void evaluateTrainImageHalfCompact(float* data,
                                              float* out,
                                              unsigned int* functionSet,
                                              cudaTextureObject_t pop,
                                              float* fitness,
                                              float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 0, 0, group_id);
    if(local_id == 0 ){
        // printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        // for(i = 0; i < 11; i ++){
            pixel = tex3D(pop, 0, 0, group_id);
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        // }
    }
    */

    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitTrainCompactImage_RG(pop, data, out, fitness);
}

__global__ void evaluateValidationImageHalfCompact(float* data,
                                                  float* out,
                                                  unsigned int* functionSet,
                                                  cudaTextureObject_t pop,
                                                  float* fitness,
                                                  float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationCompactImage_RG(pop, data, out, fitnessValidation);
}

/**COMPACTA IMAGE_RG */

/*==============================================================================================================================================================================================================*/

/**COMPACTA IMAGE_RGBA */

__global__ void evaluateTrainImageQuarterCompact(float* data,
                                                 float* out,
                                                 unsigned int* functionSet,
                                                 cudaTextureObject_t pop,
                                                 float* fitness,
                                                 float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    /*
    uint4 pixel;
    int4 coord = (100,0,0,0);

    pixel = tex3D(pop, 0, 0, group_id);
    if(local_id == 0 ){
        // printf("%d, %d\n", get_image_width(pop), get_image_height(pop));
        int i;
        // for(i = 0; i < 11; i ++){
            pixel = tex3D(pop, 1, 0, group_id);
            printf("%d pixel= %d, %d, %d, %d\n", group_id, pixel.x, pixel.y, pixel.z, pixel.w);
        // }
    }
    */

    // evaluateCircuitParallelTrainLinearImage(&pop[group_id], dataset, outputs, error);

    evaluateCircuitTrainCompactImage_RGBA(pop, data, out, fitness);
}

__global__ void evaluateValidationImageQuarterCompact(float* data,
                                                      float* out,
                                                      unsigned int* functionSet,
                                                      cudaTextureObject_t pop,
                                                      float* fitness,
                                                      float* fitnessValidation) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    evaluateCircuitValidationCompactImage_RGBA(pop, data, out, fitnessValidation);
}

/**COMPACTA IMAGE_RGBA */


/*==============================================================================================================================================================================================================*/

__global__ void testMemory(int* test) {

    extern __shared__ float error[];

    int group_id = blockIdx.x;
    int local_id = threadIdx.x;

    if (local_id == 0)
        printf("%d\n", test[group_id]);
}
