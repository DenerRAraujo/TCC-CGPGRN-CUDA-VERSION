//
// Espelho em CUDA (driver API + NVRTC) de OCLConfig.cpp.
// Cada metodo corresponde ao metodo de mesmo nome da versao OpenCL e faz a
// mesma sequencia de operacoes, para que a comparacao de tempo entre os dois
// frameworks reflita apenas a plataforma.
//

#include "CUDAConfig.h"

CUDAConfig::CUDAConfig() {
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, GPU_DEVICE));

    /* Contexto primario do dispositivo (equivalente ao cl::Context). */
    checkError(cuDevicePrimaryCtxRetain(&context, device));
    checkError(cuCtxSetCurrent(context));

    /* Eventos de profiling (o OpenCL usa CL_QUEUE_PROFILING_ENABLE). */
    checkError(cuEventCreate(&e_tempo_start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_end, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_train_start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_train_end, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_valid_start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_valid_end, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_test_start, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&e_tempo_test_end, CU_EVENT_DEFAULT));

    /* Equivalente a CL_DEVICE_MAX_WORK_GROUP_SIZE. */
    int maxThreads = 0;
    checkError(cuDeviceGetAttribute(&maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    maxLocalSize = (size_t)maxThreads;

    printCudaDeviceInfo();
}

void CUDAConfig::checkError(CUresult result){
    if(result != CUDA_SUCCESS){
        const char* name = nullptr;
        const char* desc = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &desc);
        std::cerr << (name ? name : "CUDA_ERROR") << ": " << (desc ? desc : "") << std::endl;
    }
}

void CUDAConfig::checkNvrtc(nvrtcResult result){
    if(result != NVRTC_SUCCESS)
        std::cerr << nvrtcGetErrorString(result) << std::endl;
}

void CUDAConfig::printCudaDeviceInfo(){
    char name[256];
    int major = 0, minor = 0, sms = 0, maxThreads = 0, sharedMem = 0, constMem = 0;
    size_t totalMem = 0;
    cuDeviceGetName(name, sizeof(name), device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    cuDeviceGetAttribute(&maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    cuDeviceGetAttribute(&sharedMem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    cuDeviceGetAttribute(&constMem, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
    cuDeviceTotalMem(&totalMem, device);

    std::cout << "Available CUDA Device:\n";
    std::cout << "[" << GPU_DEVICE << "]" << name << std::endl;
    std::cout << "\tCompute Capability: " << major << "." << minor << std::endl;
    std::cout << "\tMax Comp Un: " << sms << std::endl;
    std::cout << "\tMax WrkGr Sz: " << maxThreads << std::endl;
    std::cout << "\tGlobal Mem Size: " << totalMem << std::endl;
    std::cout << "\tLocal Mem Size: " << sharedMem << std::endl;
    std::cout << "\tMax Const Size: " << constMem << std::endl;

    std::cout << "\tNode size: " << sizeof(Node) << std::endl;
    std::cout << "\tNode size: " << sizeof(CompactNode) << std::endl;
    std::cout << "\tIndividuals size: " << sizeof(Chromosome) << std::endl;
    std::cout << "\tIndividuals size: " << sizeof(CompactChromosome) << std::endl;
    std::cout <<  sizeof(uint16_t) << " " << sizeof(int) << " " << sizeof(unsigned int) << " " << sizeof(float) << " " << sizeof(double) <<std::endl;
}

void CUDAConfig::allocateBuffers(Parameters* p, int sizeTrain, int sizeValid, int sizeTest){
    std::cout << "Allocating buffers... " << std::endl;

    ///Buffers
    checkError(cuMemAlloc(&bufferSeeds, NUM_INDIV * maxLocalSize  * sizeof(int)));

    checkError(cuMemAlloc(&bufferDatasetTrain, sizeTrain * p->N * sizeof(float)));
    checkError(cuMemAlloc(&bufferOutputsTrain, sizeTrain * p->O * sizeof(float)));

    checkError(cuMemAlloc(&bufferDatasetValid, sizeValid * p->N * sizeof(float)));
    checkError(cuMemAlloc(&bufferOutputsValid, sizeValid * p->O * sizeof(float)));

    checkError(cuMemAlloc(&bufferDatasetTest, sizeTest * p->N * sizeof(float)));
    checkError(cuMemAlloc(&bufferOutputsTest, sizeTest * p->O * sizeof(float)));

    checkError(cuMemAlloc(&bufferFunctions, ((p->NUM_FUNCTIONS)) * sizeof(unsigned int)));

    checkError(cuMemAlloc(&bufferBest, sizeof(Chromosome)));
    checkError(cuMemAlloc(&bufferPopulation, NUM_INDIV * sizeof(Chromosome)));
    checkError(cuMemAlloc(&bufferPopulationCompact, NUM_INDIV * sizeof(CompactChromosome)));

    checkError(cuMemAlloc(&bufferFitness, NUM_INDIV * sizeof(float)));
    checkError(cuMemAlloc(&bufferFitnessValidation, NUM_INDIV * sizeof(float)));

    numPointsTrain = sizeTrain;
    numPointsValid = sizeValid;
    numPointsTest = sizeTest;
    numPoints = numPointsTrain > numPointsValid ? numPointsTrain : numPointsValid;

    (transposeDatasetTrain) = new float [numPointsTrain * p->N];
    (transposeOutputsTrain) = new float [numPointsTrain * p->O];

    (transposeDatasetValid) = new float [numPointsValid * p->N];
    (transposeOutputsValid) = new float [numPointsValid * p->O];

    (transposeDatasetTest) = new float [numPointsTest * p->N];
    (transposeOutputsTest) = new float [numPointsTest * p->O];
}

void CUDAConfig::releaseAll() {
    /* Igual ao OpenCL: a liberacao explicita esta desativada. */
    std::cout << "All buffers released" << std::endl;
}

void CUDAConfig::setNDRages() {
    //FOR GPU

    std::cout << "Setting NDRanges ..." << std::endl;
    if(numPointsTrain < maxLocalSize){
        localSizeTrain = numPointsTrain;
    }else{
        localSizeTrain = maxLocalSize;
    }

    if(numPointsValid < maxLocalSize){
        localSizeValid = numPointsValid;
    }else{
        localSizeValid = maxLocalSize;
    }

    if(numPointsTest < maxLocalSize){
        localSizeTest = numPointsTest;
    }else{
        localSizeTest = maxLocalSize;
    }

    if(numPoints < maxLocalSize){
        localSizeAval = numPoints;
    }else{
        localSizeAval = maxLocalSize;
    }

    // One individual per work-group (= um individuo por bloco)
    globalSizeTrain = localSizeTrain * NUM_INDIV;
    globalSizeValid = localSizeValid * NUM_INDIV;
    globalSizeTest = localSizeTest * 1;
    globalSizeAval = localSizeAval * NUM_INDIV;

    if(MAX_NODES * MAX_ARITY < maxLocalSize){
        localSizeEvol = MAX_NODES * MAX_ARITY;
    }else{
        localSizeEvol = maxLocalSize;
    }

    globalSizeEvol = localSizeEvol * NUM_INDIV;
}

void CUDAConfig::setCompileFlags(){

    std::cout << "Setting compile flags..." << std::endl;

    if( !IsPowerOf2( localSizeAval ) )
        compileFlags.push_back("-DLOCAL_SIZE_IS_NOT_POWER_OF_2");
    if( !IsPowerOf2( localSizeTrain ) )
        compileFlags.push_back("-DLOCAL_SIZE_TRAIN_IS_NOT_POWER_OF_2");
    if( !IsPowerOf2( localSizeValid ) )
        compileFlags.push_back("-DLOCAL_SIZE_VALIDATION_IS_NOT_POWER_OF_2");
    if( !IsPowerOf2( localSizeTest ) )
        compileFlags.push_back("-DLOCAL_SIZE_TEST_IS_NOT_POWER_OF_2");

    if( numPoints % (localSizeAval) != 0 )
        compileFlags.push_back("-DNUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE");
    if( numPointsValid % (localSizeAval) != 0 )
        compileFlags.push_back("-DNUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE_GLOBAL");

    if( numPointsTrain % (localSizeTrain) != 0 )
        compileFlags.push_back("-DNUM_POINTS_TRAIN_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE");
    if( numPointsValid % (localSizeValid) != 0 )
        compileFlags.push_back("-DNUM_POINTS_VALIDATION_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE");
    if( numPointsTest % (localSizeTest) != 0 )
        compileFlags.push_back("-DNUM_POINTS_TEST_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE");

    compileFlags.push_back("-DLOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeAval) ));
    compileFlags.push_back("-DLOCAL_SIZE_TRAIN_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeTrain) ));
    compileFlags.push_back("-DLOCAL_SIZE_VALIDATION_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeValid) ));
    compileFlags.push_back("-DLOCAL_SIZE_TEST_ROUNDED_UP_TO_POWER_OF_2="
                    + ToString( NextPowerOf2(localSizeTest) ));
}

std::string CUDAConfig::setProgramSource(Parameters* p, Dataset* fullData){
    /* Mesmo trabalho e mesmo cabecalho de #define da versao OpenCL
       (datasetString/outputString sao montadas mas nao usadas, como la). */
    float* transposeDataset = new float [fullData->M * fullData->N];
    float* transposeOutput = new float [fullData->M * fullData->O];
    transposeData(fullData, &transposeDataset, &transposeOutput);

    std::string datasetString = "{";
    int i = 0;
    for(i = 0; i < fullData->M-1; i++){
        datasetString+= ToString(transposeDataset[i]) + ", ";
    }
    datasetString+= ToString(transposeDataset[i]);
    datasetString += "};";

    std::string outputString = "{";
    i = 0;
    for(i = 0; i < fullData->M-1; i++){
        outputString+= ToString(transposeOutput[i]) + ", ";
    }
    outputString+= ToString(transposeOutput[i]);
    outputString += "};";

    std::string program_src =
            "#define SEED "  + ToString( SEED ) + "\n" +
            "#define N   " + ToString( p->N ) + "\n" +
            "#define O   " + ToString( p->O ) + "\n" +
            "#define M   " + ToString( numPoints ) + "\n" +
            "#define M_TRAIN   " + ToString( numPointsTrain ) + "\n" +
            "#define M_VALIDATION   " + ToString( numPointsValid ) + "\n" +
            "#define M_TEST   " + ToString( numPointsTest ) + "\n" +
            "#define WEIGTH_RANGE    " + ToString(p->weightRange) + "\n" +
            "#define NUM_FUNCTIONS    " + ToString(p->NUM_FUNCTIONS) + "\n" +
            "#define SIG    " + ToString(SIG) + "\n" +
            "#define NAND    " + ToString(NAND) + "\n" +
            "#define AND    " + ToString(AND) + "\n" +
            "#define OR    " + ToString(OR) + "\n" +
            "#define NOR    " + ToString(NOR) + "\n" +
            "#define NOT    " + ToString(NOT) + "\n" +
            "#define XOR    " + ToString(XOR) + "\n" +
            "#define XNOR    " + ToString(XNOR) + "\n" +
            "#define MAX_NODES     " + ToString( MAX_NODES ) + "\n" +
            "#define MAX_OUTPUTS  " + ToString( MAX_OUTPUTS ) + "\n" +
            "#define NUM_INDIV   " + ToString( NUM_INDIV ) + "\n" +
            "#define MAX_ARITY "    + ToString( MAX_ARITY ) + "\n" +
            "#define PROB_CROSS  "+ ToString( PROB_CROSS ) + "\n" +
            "#define PROB_MUT    "+ ToString( PROB_MUT ) + "\n" +
            "#define NUM_GENERATIONS    "+ ToString( NUM_GENERATIONS ) + "\n" +
            "#define CONST_PI    "+ ToString( CONST_PI ) + "\n" +
            "#define LOCAL_SIZE " + ToString( localSizeAval ) + "\n"+
            "#define LOCAL_SIZE_TRAIN " + ToString( localSizeTrain ) + "\n"+
            "#define LOCAL_SIZE_VALIDATION " + ToString( localSizeValid ) + "\n"+
            "#define LOCAL_SIZE_TEST " + ToString( localSizeTest ) + "\n" +
            "#define LOCAL_SIZE_EVOL " + ToString( localSizeEvol ) + "\n";

    delete [] transposeDataset;
    delete [] transposeOutput;

    return program_src;
}

void CUDAConfig::buildProgram(Parameters* p, Dataset* fullData, std::string sourceFileStr){
    ///Program build (JIT, como cl::Program::build)
    std::ifstream sourceFileName(sourceFileStr);
    std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName),(std::istreambuf_iterator<char>()));

    std::string program_src = setProgramSource(p,fullData) + sourceFile;

    nvrtcProgram prog;
    checkNvrtc(nvrtcCreateProgram(&prog, program_src.c_str(), "kernel.cu", 0, nullptr, nullptr));

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    /* Equivalentes de "-cl-no-signed-zeros -cl-fast-relaxed-math" e "-I ./kernels". */
    compileFlags.push_back("--use_fast_math");
    compileFlags.push_back("--include-path=./kernels");
    /* Uma GPU mais nova que o toolkit (ex.: RTX 50xx, compute 12.0, com NVRTC 12.4)
     * nao tem a sua arquitetura reconhecida pelo NVRTC. Nesse caso usa a maior
     * arquitetura suportada abaixo da GPU: o driver faz o JIT do PTX para ela. */
    int deviceArch = major * 10 + minor;
    int targetArch = deviceArch;
    int numArchs = 0;
    if(nvrtcGetNumSupportedArchs(&numArchs) == NVRTC_SUCCESS && numArchs > 0){
        std::vector<int> archs(numArchs);
        if(nvrtcGetSupportedArchs(archs.data()) == NVRTC_SUCCESS){
            targetArch = 0;
            for(int a : archs)
                if(a <= deviceArch && a > targetArch)
                    targetArch = a;
            if(targetArch == 0)
                targetArch = deviceArch;
        }
    }
    if(targetArch != deviceArch)
        std::cerr << "Aviso: NVRTC nao suporta compute_" << deviceArch
                  << "; usando compute_" << targetArch << " (PTX compilado via JIT pelo driver)." << std::endl;
    compileFlags.push_back("--gpu-architecture=compute_" + ToString(targetArch));

    std::vector<const char*> opts;
    for(size_t i = 0; i < compileFlags.size(); i++)
        opts.push_back(compileFlags[i].c_str());

    nvrtcResult result = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

    if(result != NVRTC_SUCCESS){
        std::cerr << nvrtcGetErrorString(result) << std::endl;
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string buildLog(logSize, '\0');
        nvrtcGetProgramLog(prog, &buildLog[0]);
        std::cerr << "Build log:" << std::endl
                  << buildLog << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t ptxSize = 0;
    checkNvrtc(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptx(ptxSize, '\0');
    checkNvrtc(nvrtcGetPTX(prog, &ptx[0]));
    nvrtcDestroyProgram(&prog);

    checkError(cuModuleLoadDataEx(&program, ptx.c_str(), 0, nullptr, nullptr));
}

void CUDAConfig::buildKernels(){
    ///Kernels
    /* No CUDA os argumentos sao passados no lancamento (ver enqueue*), nao aqui. */
#if DEFAULT
    checkError(cuModuleGetFunction(&kernelEvaluate, program, "evaluateTrainValidation"));
    checkError(cuModuleGetFunction(&kernelTrain, program, "evaluateTrain"));
    checkError(cuModuleGetFunction(&kernelValid, program, "evaluateValidation"));
#elif COMPACT
    checkError(cuModuleGetFunction(&kernelTrainCompact, program, "evaluateTrainCompact"));
    checkError(cuModuleGetFunction(&kernelValidCompact, program, "evaluateValidationCompact"));
#endif
    checkError(cuModuleGetFunction(&kernelTest, program, "evaluateTest"));
    checkError(cuModuleGetFunction(&kernelEvolve, program, "evolve"));
}

void CUDAConfig::transposeDatasets(Dataset* train, Dataset* valid, Dataset* test){
    std::cout << "Transpondo dados..." << std::endl;
    transposeData(train, &transposeDatasetTrain, &transposeOutputsTrain);
    transposeData(valid, &transposeDatasetValid, &transposeOutputsValid);
    transposeData(test, &transposeDatasetTest, &transposeOutputsTest);
}

void CUDAConfig::transposeDatasets(Dataset* train){
    std::cout << "Transpondo dados..." << std::endl;
    transposeData(train, &transposeDatasetTrain, &transposeOutputsTrain);
}

void CUDAConfig::writeReadOnlyBufers(Parameters* p){
    checkError(cuMemcpyHtoDAsync(bufferDatasetTrain, transposeDatasetTrain, numPointsTrain * p->N * sizeof(float), 0));
    checkError(cuMemcpyHtoDAsync(bufferOutputsTrain, transposeOutputsTrain, numPointsTrain * p->O * sizeof(float), 0));

    checkError(cuMemcpyHtoDAsync(bufferDatasetValid, transposeDatasetValid, numPointsValid * p->N * sizeof(float), 0));
    checkError(cuMemcpyHtoDAsync(bufferOutputsValid, transposeOutputsValid, numPointsValid * p->O * sizeof(float), 0));

    checkError(cuMemcpyHtoDAsync(bufferDatasetTest, transposeDatasetTest, numPointsTest * p->N * sizeof(float), 0));
    checkError(cuMemcpyHtoDAsync(bufferOutputsTest, transposeOutputsTest, numPointsTest * p->O * sizeof(float), 0));

    checkError(cuMemcpyHtoDAsync(bufferFunctions, p->functionSet, (p->NUM_FUNCTIONS) * sizeof(unsigned int), 0));

    checkError(cuCtxSynchronize());
}

void CUDAConfig::transposeData(Dataset* data, float** transposeDataset, float** transposeOutputs){
    //transposição necessária para otimizar a execução com acessos sequenciais à memória
    unsigned int pos = 0;
    for(int j = 0; j < data->N; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeDataset)[pos++] = data->data[i][j];
        }
    }

    pos = 0;
    for(int j = 0; j < data->O; ++j ){
        for(int i = 0; i < data->M; ++i ){
            (*transposeOutputs)[pos++] = data->output[i][j];
        }
    }
}

void CUDAConfig::writeBestBuffer(Chromosome* best){
    checkError(cuMemcpyHtoDAsync(bufferBest, best, sizeof(Chromosome), 0));
}

void CUDAConfig::writePopulationBuffer(Chromosome* population){
    checkError(cuMemcpyHtoDAsync(bufferPopulation, population, NUM_INDIV * sizeof(Chromosome), 0));
}

void CUDAConfig::writePopulationCompactBuffer(CompactChromosome* population){
    checkError(cuMemcpyHtoDAsync(bufferPopulationCompact, population, NUM_INDIV * sizeof(CompactChromosome), 0));
}

void CUDAConfig::readBestBuffer(Chromosome* best){
    checkError(cuMemcpyDtoHAsync(best, bufferBest, sizeof(Chromosome), 0));
}

void CUDAConfig::readPopulationBuffer(Chromosome* population){
    checkError(cuMemcpyDtoHAsync(population, bufferPopulation, NUM_INDIV * sizeof(Chromosome), 0));
}

void CUDAConfig::readPopulationCompactBuffer(CompactChromosome* population){
    checkError(cuMemcpyDtoHAsync(population, bufferPopulationCompact, NUM_INDIV * sizeof(CompactChromosome), 0));
}

void CUDAConfig::readFitnessBuffer(){
    checkError(cuMemcpyDtoHAsync(fitness, bufferFitness, NUM_INDIV * sizeof(float), 0));
}

void CUDAConfig::readFitnessValidationBuffer(){
    checkError(cuMemcpyDtoHAsync(fitnessValidation, bufferFitnessValidation, NUM_INDIV * sizeof(float), 0));
}

void CUDAConfig::readSeedsBuffer(int* seeds){
    checkError(cuMemcpyDtoHAsync(seeds, bufferSeeds, NUM_INDIV * maxLocalSize  * sizeof(int), 0));
}

void CUDAConfig::finishCommandQueue(){
    checkError(cuCtxSynchronize());
}

/* Equivalente a enqueueNDRangeKernel(kernel, NullRange, NDRange(global), NDRange(local), nullptr, &evento). */
void CUDAConfig::launch(CUfunction f, size_t globalSize, size_t localSize, unsigned int sharedBytes,
                        void** args, CUevent start, CUevent end){
    checkError(cuEventRecord(start, 0));
    checkError(cuLaunchKernel(f,
                              (unsigned int)(globalSize / localSize), 1, 1,
                              (unsigned int)localSize, 1, 1,
                              sharedBytes, 0, args, nullptr));
    checkError(cuEventRecord(end, 0));
}

void CUDAConfig::enqueueTrainKernel(){
    void* args[] = { &bufferDatasetTrain, &bufferOutputsTrain, &bufferFunctions, &bufferPopulation,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelTrain, globalSizeTrain, localSizeTrain, (unsigned int)(localSizeTrain * sizeof(float)),
           args, e_tempo_train_start, e_tempo_train_end);
}

void CUDAConfig::enqueueValidationKernel(){
    void* args[] = { &bufferDatasetValid, &bufferOutputsValid, &bufferFunctions, &bufferPopulation,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelValid, globalSizeValid, localSizeValid, (unsigned int)(localSizeValid * sizeof(float)),
           args, e_tempo_valid_start, e_tempo_valid_end);
}

void CUDAConfig::enqueueTestKernel(){
    void* args[] = { &bufferDatasetTest, &bufferOutputsTest, &bufferFunctions, &bufferBest,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelTest, globalSizeTest, localSizeTest, (unsigned int)(localSizeTest * sizeof(float)),
           args, e_tempo_test_start, e_tempo_test_end);
}

void CUDAConfig::enqueueTrainCompactKernel(){
    void* args[] = { &bufferDatasetTrain, &bufferOutputsTrain, &bufferFunctions, &bufferPopulationCompact,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelTrainCompact, globalSizeTrain, localSizeTrain, (unsigned int)(localSizeTrain * sizeof(float)),
           args, e_tempo_train_start, e_tempo_train_end);
}

void CUDAConfig::enqueueValidationCompactKernel(){
    void* args[] = { &bufferDatasetValid, &bufferOutputsValid, &bufferFunctions, &bufferPopulationCompact,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelValidCompact, globalSizeValid, localSizeValid, (unsigned int)(localSizeValid * sizeof(float)),
           args, e_tempo_valid_start, e_tempo_valid_end);
}

void CUDAConfig::enqueueEvolveKernel(){
    void* args[] = { &bufferFunctions, &bufferSeeds, &bufferPopulation, &bufferBest };
    launch(kernelEvolve, globalSizeEvol, localSizeEvol, 0, args, e_tempo_start, e_tempo_end);
}

void CUDAConfig::enqueueEvaluationKernel(){
    void* args[] = { &bufferDatasetTrain, &bufferOutputsTrain, &bufferDatasetValid, &bufferOutputsValid,
                     &bufferFunctions, &bufferPopulation, &bufferBest,
                     &bufferFitness, &bufferFitnessValidation };
    launch(kernelEvaluate, globalSizeAval, localSizeAval, (unsigned int)(localSizeAval * sizeof(float)),
           args, e_tempo_start, e_tempo_end);
}

/* Tempos em segundos, como na versao OpenCL ((fim - inicio) / 1e9). */
static double elapsedSeconds(CUevent start, CUevent end){
    float ms = 0;
    cuEventSynchronize(end);
    cuEventElapsedTime(&ms, start, end);
    return ms / 1.0E3;
}

double CUDAConfig::getKernelElapsedTime(){
    return elapsedSeconds(e_tempo_start, e_tempo_end);
}
double CUDAConfig::getKernelElapsedTimeTrain(){
    return elapsedSeconds(e_tempo_train_start, e_tempo_train_end);
}
double CUDAConfig::getKernelElapsedTimeValid(){
    return elapsedSeconds(e_tempo_valid_start, e_tempo_valid_end);
}
double CUDAConfig::getKernelElapsedTimeTest(){
    return elapsedSeconds(e_tempo_test_start, e_tempo_test_end);
}

void CUDAConfig::compactChromosome(Chromosome* population, CompactChromosome* compactPopulation){
    //union IntFloat int_float;
    for(int i = 0; i < NUM_INDIV; i++ ) {
        for(int j = 0; j < MAX_NODES; j++){
            if(j%2 == 0) {
                //std::cout << population[i].activeNodes[j] << " " << population[i].activeNodes[j + 1] << std::endl;
                compactPopulation[i].activeNodes[j / 2] =
                        return_compact_inputs(
                                population[i].activeNodes[j],
                                population[i].activeNodes[j + 1]
                        );

                //std::cout << compactPopulation[i].activeNodes[j / 2] << std::endl;
               // std::cout << (compactPopulation[i].activeNodes[j / 2] >> 16) << " "
                //          << (compactPopulation[i].activeNodes[j / 2] & 0xFFFF) << std::endl << std::endl;
            }
            compactPopulation[i].nodes[j].function_inputs_active =
                    return_function_inputs_active(
                            population[i].nodes[j].function,
                            population[i].nodes[j].maxInputs,
                            population[i].nodes[j].active
                            );

            for(int k = 0; k < MAX_ARITY; k+=2){
                compactPopulation[i].nodes[j].inputs[k/2] =
                        return_compact_inputs(
                                population[i].nodes[j].inputs[k],
                                population[i].nodes[j].inputs[k+1]
                                );

                compactPopulation[i].nodes[j].inputsWeight[k] = population[i].nodes[j].inputsWeight[k];
                compactPopulation[i].nodes[j].inputsWeight[k+1] = population[i].nodes[j].inputsWeight[k+1];


                /*
                compactPopulation[i].nodes[j].inputsWeight[k/2] =
                        return_compact_inputs_weights(
                                population[i].nodes[j].inputsWeight[k],
                                population[i].nodes[j].inputsWeight[k+1]
                        );
                */
                /*unsigned int aux = compactPopulation[i].nodes[j].inputsWeight[k/2] & (0xFFFF0000);
                std::cout << *(float*)(&aux) << " " << population[i].nodes[j].inputsWeight[k] << std::endl;
                aux = compactPopulation[i].nodes[j].inputsWeight[k/2] << 16;
                std::cout << *(float*)(&aux) << " " << population[i].nodes[j].inputsWeight[k+1] << std::endl;
*/
                /*int_float.ui = compactPopulation[i].nodes[j].inputsWeight[k/2] & (0xFFFF0000);
                std::cout << int_float.f << " " << population[i].nodes[j].inputsWeight[k] << std::endl;
                int_float.ui = (compactPopulation[i].nodes[j].inputsWeight[k/2] << 16);
                std::cout << int_float.f<< " " << population[i].nodes[j].inputsWeight[k+1] << std::endl;
                */
            }
        }
        for(int j = 0; j < MAX_OUTPUTS; j++) {
            /*unsigned int in0 = population[i].output[j];
            unsigned int in1;
            if(j+1 < MAX_OUTPUTS){
                in1 = population[i].output[j+1];
            } else {
                in1 = (0xFFFF);
            }*/
            compactPopulation[i].output[j] = population[i].output[j];
        }
        compactPopulation[i].numActiveNodes = population[i].numActiveNodes;
    }
}



unsigned int CUDAConfig::return_function_inputs_active(unsigned int function, unsigned int inputs, unsigned int active){
    return ( (function << 17) | (inputs << 1) | active );
}

unsigned int CUDAConfig::return_compact_inputs(unsigned int in0, unsigned int in1){
    return ((in0 << 16) | in1);
}

union IntFloat
{
    unsigned int ui;
    float f;
};

unsigned int CUDAConfig::return_compact_inputs_weights(float in0, float in1){
    //union IntFloat intf, intf2;
    //intf.f = in0;
    //intf2.f = in1;
    //std::cout << (intf2.ui >> 16) << std::endl;
    //return (intf.ui & (0xFFFF0000)) | (intf2.ui >> 16);
    return  (*(unsigned int*)(&in0) & (0xFFFF0000)) |  (*(unsigned int*)(&in1) >>16 );
}
