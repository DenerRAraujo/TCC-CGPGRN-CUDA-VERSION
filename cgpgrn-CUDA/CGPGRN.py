import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import copy
from sklearn.metrics import silhouette_score
import numpy as np
import argparse
from itertools import product, permutations, combinations, combinations_with_replacement
import glob
import sys
from warnings import simplefilter
import subprocess
from multiprocessing import Pool, freeze_support
from time import perf_counter
import math

sys.path.insert(0, os.curdir + '/include')

import preProcessing as PP
import clusterData as CD
import discretizeData as DD
import generateOutputs as GO
import utils as Utils
import makeFile as MKFile
import objects as objs

simplefilter(action='ignore', category=FutureWarning)


def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(description='Complete pre-processing and script generator for CGPGRN framework (CUDA Enabled).')

    parser.add_argument('-cm', dest='clusterMethod', type=str, default='None',
                        help='Cluster method used for grouping genes.')
    parser.add_argument('-nc', dest='n_clusters', type=str, default='None',
                        help='The maximum number of clusters to be tested with clustering methods.')
    parser.add_argument('-t', dest='correlation_threshold', type=str, default='None',
                        help='The correlation coefficient threshold for correlation methods.')
    parser.add_argument('-e', dest='expressionDataFile', type=str, default='ExpressionData.csv',
                        help='Path to expression data file.')
    parser.add_argument('-p', dest='pseudotimeFile', type=str, default='PseudoTime.csv',
                        help='Path to pseudotime file.')
    parser.add_argument('-pn', '--problemName', dest='problemName', type=str, default='None',
                        help='Problem Name.')
    parser.add_argument('-s', '--suffix', dest='suffix', type=str, default='None',
                        help='Suffix for generated files.')
    parser.add_argument('-d', '--discretization', dest='discretizationApproach', type=str, default='None',
                        help='Discretization approach for spline data.')
    parser.add_argument('-r', '--independentRuns', dest='independentRuns', type=int, default=1,
                        help='Number of independent runs of CGP.')
    parser.add_argument('-fd', dest='fullDiscretization', action='store_true', default=False,
                        help='Discretize every spline generated file.')
    parser.add_argument('-sf', '--splineFile', dest='argSplineFile', type=str, default='None',
                        help='Spline File.')
    parser.add_argument('-sl', dest='splineList', action='store_true', default=False,
                        help='When True, -sf must be a txt file.')
    parser.add_argument('-run', dest='runCGP', action='store_true', default=False,
                        help='Automatically run CGPGRN inference procedure after preprocessing.')
    parser.add_argument('-kd', dest='keepData', action='store_true', default=False,
                        help='Keep Source and Spline Data after complete pipeline execution.')
    parser.add_argument('-fullTT', dest='fullTT', action='store_true', default=False,
                        help='Use full truth tables for clustering methods.')
    parser.add_argument('-gsl', dest='generateSplineList', action='store_true', default=False,
                        help='Generate spline list file.')
    parser.add_argument('-cgpn', '--nodes', dest='cgpNodes', type=str, default='None',
                        help='The number of nodes to encode CGPGRN genotype.')
    parser.add_argument('-cgpg', '--gens', dest='cgpGens', type=str, default='None',
                        help='The maximum number of generations for CGPGRN search.')

    # NOVA ADIÇÃO: Argumento para ativar execução em CUDA/GPU
    parser.add_argument('-cuda', dest='useCUDA', action='store_true', default=True,
                        help='Usar aceleracao via NVIDIA CUDA. Padrao: True.')

    return parser


def parse_arguments():
    parser = get_parser()
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':

    t0_General = perf_counter()

    correlationMethods = Utils.correlationMethods
    clusteringMethods = Utils.clusteringMethods
    possibleDiscretizationApproaches = Utils.possibleDiscretizationApproaches

    opts = parse_arguments()

    logFileName = "LogFile_" + vars(opts)['problemName'] + ".txt"
    Log = Utils.Log(vars(opts), logFileName)
    Log.register('initial')

    Utils.verifyArgs(opts, Log)

    CGPGRN = objs.CGPGRN(opts, Log)
    splineDataNames = CGPGRN.performSplineStep()

    problemName = opts.problemName
    suffix = opts.suffix
    clusterMethod = opts.clusterMethod
    numberOfClusters = opts.n_clusters
    correlationThreshold = opts.correlation_threshold
    expressionDataFile = opts.expressionDataFile
    pseudotimeFile = opts.pseudotimeFile
    argSplineFile = opts.argSplineFile
    discretizationApproach = opts.discretizationApproach
    independentRuns = opts.independentRuns
    fullDiscretization = opts.fullDiscretization
    splineList = opts.splineList
    runCGP = opts.runCGP
    keepData = opts.keepData
    fullTT = opts.fullTT
    generateSplineList = opts.generateSplineList
    cgpNodes = opts.cgpNodes
    cgpGens = opts.cgpGens
    useCUDA = opts.useCUDA

    # Definir o nome do binario de acordo com o OS e suporte a CUDA
    if os.name == 'nt':
        progBinary = "./progCUDA.exe" if useCUDA else "./progW.exe"
    else:
        progBinary = "./progCUDA" if useCUDA else "./progL"

    all_SPFiles = []
    t0_spline = perf_counter()

    tf_spline = perf_counter()
    Log.register('time', str(tf_spline - t0_spline))
    Log.register('message', "End of spline step.")
    print("End of spline step.")

    all_SPFiles.append(tf_spline - t0_spline)

    mainDirectory = os.getcwd()

    all_discretizedFiles = []
    all_genesNamesFiles = []
    all_directories = []
    all_dirExecutions = []
    additional_discretizedFiles = []

    all_CDTimes = []
    all_DDTimes = []

    print("Processing spline data...")
    Log.register('message', "Processing spline data...")

    for splineDataName in splineDataNames:
        splitName = splineDataName.split("_")
        currentPseudoTime = splitName[len(splitName)-1].replace('.csv', '')

        print("Current pseudotime: ", currentPseudoTime)
        Log.register('message', "Current pseudotime " + str(currentPseudoTime))
        print("Starting clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('message', "Starting clustering step for pseudotime " + str(currentPseudoTime))

        t0_CD = perf_counter()
        dirName, bestNClusters = CGPGRN.performClusteringStep(splineDataName, currentPseudoTime)
        tf_CD = perf_counter()

        print("End clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('message', "End clustering step for pseudotime " + str(currentPseudoTime))
        Log.register('time', str(tf_CD - t0_CD))

        all_CDTimes.append(tf_CD-t0_CD)

        discretizedFiles = []
        genesNamesFiles = []
        directories = []

        t0_DD = perf_counter()
        print("Discretizing data using " + str(discretizationApproach) + " for pseudotime " + str(currentPseudoTime))
        Log.register('message', "Discretizing data using " + str(discretizationApproach) + " for pseudotime " + str(currentPseudoTime))

        it_genesNamesFiles, it_directories, it_additional_discretizedFiles, it_discretizedFiles = CGPGRN.performDiscretizationStep(dirName, bestNClusters, currentPseudoTime, splineDataName)

        for it in it_genesNamesFiles:
            genesNamesFiles.append(it)
        for it in it_directories:
            directories.append(it)
        for it in it_additional_discretizedFiles:
            additional_discretizedFiles.append(it)
        for it in it_discretizedFiles:
            discretizedFiles.append(it)

        print("Data discretization successfully performed.")
        Log.register('message', "Data discretization sucessfully performed.")

        tf_DD = perf_counter()
        Log.register('time', str(tf_DD-t0_DD))
        all_DDTimes.append(tf_DD-t0_DD)

        for DFs in discretizedFiles:
            all_discretizedFiles.append(DFs)
        for GNFs in genesNamesFiles:
            all_genesNamesFiles.append(GNFs)
        for DIRS in directories:
            all_directories.append(DIRS)

        print("Generating truth tables...")
        Log.register('message', 'Generating truth tables...')

        all_GOTimes = []

        for i in range(len(directories)):
           Log.register('message', "Generating truth tables for directory " + str(i+1) + "/" + str(len(directories)))
           t0_GO = perf_counter()
           truthTableDirectory = directories[i] + "/truth_tables" + "_" + currentPseudoTime + "/"
           if not(os.path.exists(truthTableDirectory)):
               os.mkdir(truthTableDirectory)
           if clusterMethod not in correlationMethods and clusterMethod != 'None':
               if discretizationApproach == 'BiKMeans':
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
               else:
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
           elif clusterMethod in correlationMethods:
               currentTargetGene = genesNamesFiles[i].split("_")[1].replace('.txt', '')
               targetGenes = [currentTargetGene]
               if discretizationApproach == 'BiKMeans':
                   GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 0, targetGenes)
               else:
                   GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 1, 0, targetGenes)
           else:
               if discretizationApproach == 'BiKMeans':
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)
               else:
                   if fullTT == True:
                       Log.register('message', "Using full truth table...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 0, truthTableDirectory, 0, 1)
                   else:
                       Log.register('message', "Using one output per gene...")
                       GO.generateOutputs(problemName, genesNamesFiles[i], discretizedFiles[i], 1, truthTableDirectory, 1, 1)

           tf_GO = perf_counter()
           Log.register('time', str(tf_GO - t0_GO))
           all_GOTimes.append(tf_GO - t0_GO)

        Utils.mkdir('executions_parallel')
        Utils.mkdir('time_counting')

        all_BashTimes = []
        all_geneNamesDirFULLTT = []
        all_geneNamesDir = []

        t0_Bash = perf_counter()

        print("Generating bash scripts...")
        Log.register('message', "Generating bash scripts...")
        bashScripts = []

        if clusterMethod in correlationMethods:
            for i in range(independentRuns):
                currentExecution = 'exe_' + str(i+1) + "_" + currentPseudoTime
                all_dirExecutions.append(currentExecution)
                Utils.mkdir('executions_parallel/' + currentExecution)
                Utils.mkdir('time_counting/' + currentExecution)
                currentOutputFileName = problemName + '_exe' + str(i+1) + "_" + currentPseudoTime + ".sh"
                bashScripts.append(currentOutputFileName)
                openOutFile = open(currentOutputFileName, 'w')
                for j in range(len(genesNamesFiles)):
                    currentGeneFileName = genesNamesFiles[j]
                    currentSeed = (i * len(genesNamesFiles)) + j
                    currentGeneDirectory = currentGeneFileName
                    currentGene = currentGeneFileName.split('_')[1].replace('.txt', '')
                    currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                    all_geneNamesDir.append(currentGeneDirectory)

                    # Uso da variável 'progBinary' adaptada para CUDA
                    outputString = f"{progBinary} {currentGeneDirectory} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} 0 \n"
                    openOutFile.write(outputString)
                openOutFile.close()

        elif clusterMethod in clusteringMethods:
            for i in range(bestNClusters):
                currentClusterDir = dirName + '/Cluster' + str(i)
                currentGeneNamesDir = currentClusterDir + '/geneNames_' + str(i) + "_" + currentPseudoTime + '.txt'
                os.chdir(currentClusterDir + '/truth_tables' + "_" + currentPseudoTime)
                allCurrentTruthTables = glob.glob('*.txt')
                os.chdir('../../..')
                for j in range(independentRuns):
                    currentExecution = 'exe_' + str(j+1) + '_Cluster' + str(i) + "_" + currentPseudoTime
                    all_dirExecutions.append(currentExecution)
                    Utils.mkdir('executions_parallel/' + currentExecution)
                    Utils.mkdir('time_counting/' + currentExecution)
                    currentOutputFileName = problemName + '_' + currentExecution + ".sh"
                    bashScripts.append(currentOutputFileName)
                    allGenesInCurrentCluster = []
                    all_geneNamesDir.append(currentGeneNamesDir)
                    openGeneFile = open(currentGeneNamesDir, "r")
                    for line in openGeneFile:
                        allGenesInCurrentCluster.append(line.strip())

                    openOutFile = open(currentOutputFileName, 'w')
                    if fullTT == False:
                        for k in range(len(allGenesInCurrentCluster)):
                            currentSeed = (j * len(allGenesInCurrentCluster)) + k
                            currentGene = allGenesInCurrentCluster[k]
                            currentTruthTableDirectory = dirName + '/Cluster' + str(i) + '/truth_tables' + "_" + currentPseudoTime
                            outputString = f"{progBinary} {currentGeneNamesDir} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} 0 \n"
                            openOutFile.write(outputString)
                    else:
                        all_geneNamesDirFULLTT.append(currentGeneNamesDir)
                        currentSeed = j + independentRuns*i
                        currentGene = 'None'
                        currentTruthTableDirectory = dirName + '/Cluster' + str(i) + '/truth_tables' + "_" + currentPseudoTime
                        outputString = f"{progBinary} {currentGeneNamesDir} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} 1 \n"
                        openOutFile.write(outputString)
                    openOutFile.close()

        else:
            for i in range(independentRuns):
                currentExecution = 'exe_' + str(i+1) + "_" + currentPseudoTime
                Utils.mkdir('executions_parallel/' + currentExecution)
                Utils.mkdir('time_counting/' + currentExecution)
                all_dirExecutions.append(currentExecution)
                currentOutputFileName = problemName + '_exe' + str(i+1) + "_" + currentPseudoTime + ".sh"
                bashScripts.append(currentOutputFileName)
                openOutFile = open(currentOutputFileName, 'w')
                if clusterMethod != 'None':
                    for j in range(len(genesNamesFiles)):
                        currentGeneFileName = genesNamesFiles[j]
                        currentSeed = (i * len(genesNamesFiles)) + j
                        currentGeneDirectory = currentGeneFileName
                        currentGene = currentGeneFileName.split('_')[1].replace('.txt', '') if fullTT == False else 'None'
                        currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                        all_geneNamesDir.append(currentGeneDirectory)
                        tt_flag = "0" if fullTT == False else "1"
                        outputString = f"{progBinary} {currentGeneDirectory} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} {tt_flag} \n"
                        openOutFile.write(outputString)
                else:
                    if fullTT == False:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            currentGeneDirectory = currentGeneFileName
                            currentLocalGenesOpen = open(currentGeneFileName, 'r')
                            for line in currentLocalGenesOpen:
                                currentGene = line.strip()
                                currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                                all_geneNamesDir.append(currentGeneDirectory)
                                outputString = f"{progBinary} {currentGeneDirectory} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} 0 \n"
                                openOutFile.write(outputString)
                    else:
                        for j in range(len(genesNamesFiles)):
                            currentGeneFileName = genesNamesFiles[j]
                            currentSeed = (i * len(genesNamesFiles)) + j
                            currentGeneDirectory = currentGeneFileName
                            currentGene = 'None'
                            all_geneNamesDir.append(currentGeneDirectory)
                            all_geneNamesDirFULLTT.append(currentGeneDirectory)
                            currentTruthTableDirectory = dirName + "/truth_tables" + "_" + currentPseudoTime
                            outputString = f"{progBinary} {currentGeneDirectory} {currentExecution} {problemName} {currentGene} {currentSeed} {currentTruthTableDirectory} 1 \n"
                            openOutFile.write(outputString)
                openOutFile.close()

        allMaxInputs = Utils.calculateMaxInputs(all_geneNamesDir)

        allCounts = []
        if len(all_geneNamesDirFULLTT) != 0:
            for localDir in all_geneNamesDirFULLTT:
                currentCount = sum(1 for _ in open(localDir, "r"))
                allCounts.append(currentCount)

        allShellScriptsFileName = problemName + "_full.sh"

        # Passa as especificações para o makeFile/Bash Script
        Utils.generateFullBashScript(allShellScriptsFileName, fullTT, bashScripts, allCounts, allMaxInputs, opts)

    outputFullShellScripts = open(allShellScriptsFileName, 'a')
    python_cmd = "python" if os.name == 'nt' else "python3"
    outputFullShellScripts.write(f"{python_cmd} include/postProcess.py\n")
    outputFullShellScripts.close()

    Log.register('message', "Bash scripts sucessfully generated.")
    tf_Bash = perf_counter()
    all_BashTimes.append(tf_Bash - t0_Bash)
    Log.register('time', str(tf_Bash - t0_Bash))

    tf_General = perf_counter()

    Log.register('final')
    Log.register('timef', str(tf_General - t0_General))

    parametersFileName = "CGPGRN_parameters.txt"
    with open(parametersFileName, "w") as parametersFile:
        parametersFile.write(str(all_discretizedFiles) + "\n")
        parametersFile.write(str(all_genesNamesFiles) + "\n")
        parametersFile.write(str(all_directories) + "\n")
        parametersFile.write(str(all_dirExecutions) + "\n")
        parametersFile.write("Additional Discretization Files: " + str(additional_discretizedFiles) + "\n")
        parametersFile.write(str(opts) + "\n")

    timeFileName = "CGPGRN_times.txt"
    with open(timeFileName, "w") as timeFile:
        timeFile.write("Spline Files: " + str(all_SPFiles) + " - Total: " + str(sum(all_SPFiles)) + "\n")
        timeFile.write("Clustering Step: " + str(all_CDTimes) + " - Total: " + str(sum(all_CDTimes)) + "\n")
        timeFile.write("Discretization: " + str(all_DDTimes) + " - Total: " + str(sum(all_DDTimes)) + "\n")
        timeFile.write("Truth Table Generation: " + str(all_GOTimes) + " - Total: " + str(sum(all_GOTimes)) + "\n")
        timeFile.write("Bash Generation: " + str(all_BashTimes) + " - Total: " + str(sum(all_BashTimes)) + "\n")
        timeFile.write("Total Elapsed Time: " + str(tf_General - t0_General) + "\n")

    print(opts)

    if runCGP:
        if os.name != 'nt':
            os.system('chmod +x *.sh')
            if os.path.exists('progCUDA'):
                os.system('chmod +x progCUDA')
            elif os.path.exists('progL'):
                os.system('chmod +x progL')
            shellFile = "./" + allShellScriptsFileName
            subprocess.call(shellFile, shell=True)
        else:
            os.popen(allShellScriptsFileName)
