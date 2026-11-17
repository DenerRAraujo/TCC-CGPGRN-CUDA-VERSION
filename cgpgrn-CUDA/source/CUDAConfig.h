//
// Espelho em CUDA de OCLConfig.h (versao OpenCL do CGPGRN).
//
// Mapeamento usado (OpenCL -> CUDA driver API + NVRTC):
//   cl::Context / CommandQueue  -> CUcontext (contexto primario) + stream padrao (0)
//   cl::Program (build JIT)     -> nvrtcProgram -> PTX -> CUmodule
//   cl::Kernel                  -> CUfunction
//   cl::Buffer                  -> CUdeviceptr
//   enqueueWrite/ReadBuffer     -> cuMemcpyHtoDAsync / cuMemcpyDtoHAsync
//   cmdQueue.finish()           -> cuCtxSynchronize()
//   profiling do evento         -> CUevent antes/depois do lancamento
//   NDRange(global, local)      -> grid = global/local blocos, local threads/bloco
//   argumento __local           -> memoria __shared__ dinamica do lancamento
//
// Os kernels sao compilados em tempo de execucao, como no OpenCL, para que as
// constantes do problema entrem como literais (#define) e as flags *_IS_NOT_*
// so sejam ligadas quando verdadeiras.
//

#ifndef P_CGPDE_CUDACONFIG_H
#define P_CGPDE_CUDACONFIG_H

#include <cuda.h>
#include <nvrtc.h>

#include "utils.h"
#include "constants.h"

#if IMAGE_R || IMAGE_RG || IMAGE_RGBA || COMPACT_R || COMPACT_RG || COMPACT_RGBA
#error "Os modos IMAGE_* / COMPACT_R* nao foram portados para CUDA. Use DEFAULT ou COMPACT."
#endif

class CUDAConfig {
public:
    CUDAConfig();

    float fitness[NUM_INDIV];
    float fitnessValidation[NUM_INDIV];

    CUdevice device;
    CUcontext context;
    CUmodule program;

    CUfunction kernelEvaluate = nullptr;
    CUfunction kernelTrain = nullptr;
    CUfunction kernelValid = nullptr;
    CUfunction kernelTest = nullptr;
    CUfunction kernelTrainCompact = nullptr;
    CUfunction kernelValidCompact = nullptr;
    CUfunction kernelEvolve = nullptr;

    CUdeviceptr bufferSeeds = 0;

    CUdeviceptr bufferDatasetTrain = 0;
    CUdeviceptr bufferOutputsTrain = 0;

    CUdeviceptr bufferDatasetValid = 0;
    CUdeviceptr bufferOutputsValid = 0;

    CUdeviceptr bufferDatasetTest = 0;
    CUdeviceptr bufferOutputsTest = 0;

    CUdeviceptr bufferFunctions = 0;

    CUdeviceptr bufferBest = 0;
    CUdeviceptr bufferPopulation = 0;
    CUdeviceptr bufferPopulationCompact = 0;

    CUdeviceptr bufferFitness = 0;
    CUdeviceptr bufferFitnessValidation = 0;

    float* transposeDatasetTrain;
    float* transposeOutputsTrain;

    float* transposeDatasetValid;
    float* transposeOutputsValid;

    float* transposeDatasetTest;
    float* transposeOutputsTest;

    size_t numPoints;
    size_t numPointsTrain;
    size_t numPointsValid;
    size_t numPointsTest;

    size_t maxLocalSize;

    size_t localSizeTrain;
    size_t localSizeValid;
    size_t localSizeTest;

    size_t globalSizeTrain;
    size_t globalSizeValid;
    size_t globalSizeTest;

    size_t localSizeAval;
    size_t globalSizeAval;

    size_t localSizeEvol;
    size_t globalSizeEvol;

    /** Opcoes passadas ao NVRTC (equivalente a string compileFlags do OpenCL). */
    std::vector<std::string> compileFlags;

    /** Eventos de profiling (equivalentes a e_tempo, e_tempo_train, ...). */
    CUevent e_tempo_start, e_tempo_end;
    CUevent e_tempo_train_start, e_tempo_train_end;
    CUevent e_tempo_valid_start, e_tempo_valid_end;
    CUevent e_tempo_test_start, e_tempo_test_end;

    void allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest);
    void releaseAll();
    void setNDRages();
    void setCompileFlags();
    std::string setProgramSource(Parameters* p, Dataset* fullData);
    void buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr);
    void buildKernels();

    void transposeDatasets(Dataset* train, Dataset* valid, Dataset* test);
    void transposeDatasets(Dataset* train);
    void writeReadOnlyBufers(Parameters* p);

    void writeBestBuffer(Chromosome* best);
    void writePopulationBuffer(Chromosome* population);
    void writePopulationCompactBuffer(CompactChromosome* population);

    void readBestBuffer(Chromosome* best);
    void readPopulationBuffer(Chromosome* population);
    void readPopulationCompactBuffer(CompactChromosome* population);
    void readFitnessBuffer();
    void readFitnessValidationBuffer();
    void readSeedsBuffer(int* seeds);

    void finishCommandQueue();

    void enqueueTrainKernel();
    void enqueueValidationKernel();
    void enqueueTestKernel();
    void enqueueTrainCompactKernel();
    void enqueueValidationCompactKernel();
    void enqueueEvolveKernel();
    void enqueueEvaluationKernel();

    double getKernelElapsedTime();
    double getKernelElapsedTimeTrain();
    double getKernelElapsedTimeValid();
    double getKernelElapsedTimeTest();

    void compactChromosome(Chromosome* population, CompactChromosome* compactPopulation);

private:
    void checkError(CUresult result);
    void checkNvrtc(nvrtcResult result);
    void printCudaDeviceInfo();
    void transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs);
    void launch(CUfunction f, size_t globalSize, size_t localSize, unsigned int sharedBytes,
                void** args, CUevent start, CUevent end);

    unsigned int return_function_inputs_active(unsigned int function, unsigned int inputs, unsigned int active);
    unsigned int return_compact_inputs(unsigned int in0, unsigned int in1);
    unsigned int return_compact_inputs_weights(float in0, float in1);
};

#endif // P_CGPDE_CUDACONFIG_H
