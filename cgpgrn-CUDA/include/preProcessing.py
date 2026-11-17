import pandas as pd
import numpy as np
from csaps import csaps
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from os.path import exists
import gc
from time import perf_counter
from scipy import stats
import os
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
from itertools import repeat

class Pseudotime:
    def __init__(self, n_pseudotimes, pseudotime_file, pseudotimes=None, ordered_pseudotimes=None):
        self.pseudotime_file = pseudotime_file
        self.n_pseudotimes = n_pseudotimes
        self.pseudotimes = pseudotimes if pseudotimes is not None else []
        self.ordered_pseudotimes = ordered_pseudotimes if ordered_pseudotimes is not None else []
        self.pseudotimesToProcess = []
        self.cells = []
        
        for n_pseudotime in range(n_pseudotimes):
            self.pseudotimes.append({})
            self.ordered_pseudotimes.append([])

    def processPseudotime(self, pseudotime):
        self.pseudotimesToProcess = [] # Limpa a lista para a iteracao atual
        allCells = list(self.pseudotime_file.T.columns)
        self.cells = allCells
        for cell in allCells:
            val = self.pseudotime_file.T[cell].iloc[pseudotime]
            if str(val) != 'nan' and not pd.isna(val):
                # Arredonda para evitar diferencas de precisao decimal float64
                currentPseudotime = round(float(val), 6)
                if currentPseudotime not in self.pseudotimes[pseudotime]:
                    self.pseudotimes[pseudotime][currentPseudotime] = [cell]
                else:
                    self.pseudotimes[pseudotime][currentPseudotime].append(cell)
        
        allPseudotimes = list(self.pseudotimes[pseudotime].keys())
        self.ordered_pseudotimes[pseudotime] = sorted(allPseudotimes)
        for pt in allPseudotimes:
            if len(self.pseudotimes[pseudotime][pt]) > 1:
                self.pseudotimesToProcess.append(pt)


class ExpressionData:
    def __init__(self, n_genes, expression_file, pseudotime_object, suffix, genes=None, cells=None):
        self.n_genes = n_genes
        self.expression_file = expression_file
        self.genes = genes if genes is not None else []
        self.cells = cells if cells is not None else []
        self.pseudotime_object = pseudotime_object
        self.suffix = suffix
        self.processedPTs = {}

    def processPTs(self, n_pseudotime):
        pt_dict = self.pseudotime_object.pseudotimes[n_pseudotime]
        
        for ptValue in self.pseudotime_object.pseudotimesToProcess:
            # Busca segura considerando arredondamentos numéricos
            if ptValue in pt_dict:
                cellsToProcess = pt_dict[ptValue]
            else:
                # Procura a chave numerica mais proxima caso haja ruida decimal
                closest_key = min(pt_dict.keys(), key=lambda k: abs(k - ptValue))
                cellsToProcess = pt_dict[closest_key]

            processedCell = []
            for cell in cellsToProcess:
                if cell == cellsToProcess[0]:
                    processedCell = list(self.expression_file[cell].iloc[:])
                else:
                    currentCellExpressionData = list(self.expression_file[cell])
                    for i in range(len(currentCellExpressionData)):
                        if currentCellExpressionData[i] != 0.0:
                            processedCell[i] = (processedCell[i] + currentCellExpressionData[i]) / 2
            self.processedPTs[ptValue] = processedCell

    def calculateSmooth(self, expressionData, timePoints):
        rs = ShuffleSplit(n_splits=10, train_size=0.1, test_size=0.9, random_state=1345540)
        smooth_errors = {}
        for train, test in rs.split(expressionData):
            train.sort()
            test.sort()
            vector_train = [[], []]
            vector_test = [[], []]
            for element in train:
                vector_train[0].append(timePoints[element])
                vector_train[1].append(expressionData[element])
            for element in test:
                vector_test[0].append(timePoints[element])
                vector_test[1].append(expressionData[element])

            smooth_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

            if len(vector_train[0]) > 1 and len(vector_train[1]) > 1:
                for smooths in smooth_values:
                    ys = csaps(vector_train[0], vector_train[1], timePoints, smooth=smooths)
                    dif_total = 0
                    for element in range(len(vector_test[0])):
                        calculated = ys[element]
                        real = vector_test[1][element]
                        dif_total += pow(real - calculated, 2)
                    if smooths not in smooth_errors.keys():
                        smooth_errors[smooths] = dif_total
                    else:
                        smooth_errors[smooths] += dif_total

                best_smooth = list(smooth_errors.keys())[np.argmin(list(smooth_errors.values()))]
            else:
                best_smooth = 0.99

        return best_smooth

    def generateSplineCurve(self, expressionData, timePoints, smoothValue, timePointsClean, geneIndex, currentPt):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        ys = csaps(timePointsClean, expressionData, timePoints, smooth=smoothValue)

        currentDict2 = {}
        currentDict2[currentGeneName] = ys

        gc.collect()
        return currentDict2

    def processData(self, n_pseudotime, geneIndex):
        currentGeneName = self.expression_file.T.columns[geneIndex]
        currentCellExpressionData = []
        currentTimePoints = []
        
        current_ordered = self.pseudotime_object.ordered_pseudotimes[n_pseudotime]
        
        for i in range(len(current_ordered)):
            pt_val = current_ordered[i]
            currentCells = self.pseudotime_object.pseudotimes[n_pseudotime][pt_val]
            if len(currentCells) == 1:
                currentED = self.expression_file[currentCells[0]].iloc[geneIndex]
                if currentED != 0:
                    currentCellExpressionData.append(currentED)
                    currentTimePoints.append(pt_val)
            else:
                if pt_val in self.processedPTs:
                    currentED = self.processedPTs[pt_val][geneIndex]
                    if currentED != 0:
                        currentCellExpressionData.append(currentED)
                        currentTimePoints.append(pt_val)

        currentDF = None
        if len(currentTimePoints) < 2:
            print("Not enough expression values for gene: ", currentGeneName)
        else:
            best_smooth = self.calculateSmooth(currentCellExpressionData, currentTimePoints)
            currentDF = self.generateSplineCurve(currentCellExpressionData, current_ordered, best_smooth, currentTimePoints, geneIndex, n_pseudotime)

        gc.collect()
        return currentDF


def preProcessData(pseudotimeFile, expressionDataFile, suffix):
    allFileNames = []

    print("Initializing...")
    print("Reading pseudotime file...")
    pt_File = pd.read_csv(pseudotimeFile, index_col=0)
    nPseudoTimes = len(pt_File.columns)

    pseudotime = Pseudotime(nPseudoTimes, pt_File)

    print("Reading expression data file...")
    ExprFile = pd.read_csv(expressionDataFile, index_col=0)

    for pts in range(nPseudoTimes):
        print(f"Processing pseudotime step {pts + 1}/{nPseudoTimes}...")
        pseudotime.processPseudotime(pts)

        ExprData = ExpressionData(len(ExprFile.T.columns), ExprFile, pseudotime, suffix)

        print("Processing expression data...")
        ExprData.processPTs(pts)

        t1_start = perf_counter()

        with Pool(mp.cpu_count()) as pool:
            M = pool.starmap(ExprData.processData, zip(repeat(pts), [i for i in range(len(ExprFile.T.columns))]))

        fileName = 'splineData_' + ExprData.suffix + "_pt" + str(pts) + '.csv'

        if os.path.exists(fileName):
            os.remove(fileName)

        for l_splineData in M:
            if l_splineData is not None:
                df2 = pd.DataFrame(data=l_splineData, index=ExprData.pseudotime_object.ordered_pseudotimes[pts])
                if os.path.exists(fileName):
                    df2.T.to_csv(fileName, mode='a', header=False)
                else:
                    df2.T.to_csv(fileName, mode='a')

        t1_stop = perf_counter()
        print("Elapsed time:", t1_stop - t1_start)

        allFileNames.append(fileName)

    return allFileNames