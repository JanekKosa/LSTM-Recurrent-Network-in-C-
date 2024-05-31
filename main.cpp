
#include <iostream>
#include <ostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <random>
#include <chrono>
#include <iterator>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdio.h>

using namespace std;

struct lstmCellOutput {
    double shortTermMemory;
    double longTermMemory;
};

struct timeStepStructure {
    double price;
    string datetime;
};

vector<timeStepStructure> trainingDataSet, normalizedTrainingDataSet, evalDataSet, normalizedEvalDataSet, normalizedPredictedData, denormalizedPredictedData;
vector<vector<timeStepStructure>> setOfSamples;
vector<vector<vector<timeStepStructure>>> setOfBatches;

struct weights {
    double forgetGateSTMemWeight = 0.5;
    double forgetGateInputWeight = 0.5;
    //----------------------------
    double inputGateSTMemWeight_1stStage = 0.5;
    double inputGateInputWeight_1stStage = 0.5;
    double inputGateSTMemWeight_2ndStage = 0.5;
    double inputGateInputWeight_2ndStage = 0.5;
    //-----------------------------
    double outputGateSTMemWeight = 0.5;
    double outputGateInputWeight = 0.5;
};

struct biases {
    double  forgetGateBias = 0;
    //------------------------
    double inputGateBias_1stStage = 0;
    double inputGateBias_2ndStage = 0;
    //------------------------
    double outputGateBias = 0;
};

struct weightGradeints {
    double forgetGateSTMemWeightGrad = 0;
    double forgetGateInputWeightGrad = 0;
    //----------------------------
    double inputGateSTMemWeight_1stStageGrad = 0;
    double inputGateInputWeight_1stStageGrad = 0;
    double inputGateSTMemWeight_2ndStageGrad = 0;
    double inputGateInputWeight_2ndStageGrad = 0;
    //-----------------------------
    double outputGateSTMemWeightGrad = 0;
    double outputGateInputWeightGrad = 0;

    void clearGradients() {
        forgetGateSTMemWeightGrad = 0;
        forgetGateInputWeightGrad = 0;
        //----------------------------
        inputGateSTMemWeight_1stStageGrad = 0;
        inputGateInputWeight_1stStageGrad = 0;
        inputGateSTMemWeight_2ndStageGrad = 0;
        inputGateInputWeight_2ndStageGrad = 0;
        //-----------------------------
        outputGateSTMemWeightGrad = 0;
        outputGateInputWeightGrad = 0;
    }
};

struct biasGradients {
    double  forgetGateBiasGrad = 0;
    //------------------------
    double inputGateBias_1stStageGrad = 0;
    double inputGateBias_2ndStageGrad = 0;
    //------------------------
    double outputGateBiasGrad = 0;

    void clearGradients() {
        forgetGateBiasGrad = 0;
        //------------------------
        inputGateBias_1stStageGrad = 0;
        inputGateBias_2ndStageGrad = 0;
        //------------------------
        outputGateBiasGrad = 0;
    }
};
//+----------------------------------------------------------------------------------+
void fillDataSetVectorFromCsv(vector<timeStepStructure>& aDataSet, string aFileName) {

    ifstream pCsvFile(aFileName); // Replace "your_excel_data.csv" with your file path

    if (!pCsvFile.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }
    string pLine, pDateTimeValue, pPriceValue;

    getline(pCsvFile, pLine);
    while (getline(pCsvFile, pLine)) {
        stringstream pSSLine(pLine);
        getline(pSSLine, pDateTimeValue, ',');
        getline(pSSLine, pPriceValue);
        timeStepStructure pNewElement;
        pNewElement.price = stod(pPriceValue);
        pNewElement.datetime = pDateTimeValue;
        aDataSet.push_back(pNewElement);
        timeStepStructure pLastEle = aDataSet.back();
    }
}
//+----------------------------------------------------------------------------------+
template <typename T>

void moveElements(T& aSource, T& aDestination, int aStartIndx, int aEndIndx) {

    int i = aStartIndx;

    while (i <= aEndIndx) {
        aDestination.push_back(aSource[i]);
        i++;
    }
}
// //+----------------------------------------------------------------------------------+
//The last element in each sample is the label for the sequence of time steps in that sample.
void divideDataSetintoSamples(vector<timeStepStructure>& aDataSet, vector<vector<timeStepStructure>>& aSampleSet, int aSampleSize) {

    int pFirstElementInSample_IndxInDataSet = 0;
    int pLastElementInSample_IndxInDataSet = aSampleSize - 1;
    
    while (pLastElementInSample_IndxInDataSet != aDataSet.size() - 1){
        vector<timeStepStructure> pSample;

        moveElements(aDataSet, pSample, pFirstElementInSample_IndxInDataSet, pLastElementInSample_IndxInDataSet);

        if (pSample.size() >= 2) aSampleSet.push_back(pSample);

        pFirstElementInSample_IndxInDataSet = pLastElementInSample_IndxInDataSet + 1;
        pLastElementInSample_IndxInDataSet = pFirstElementInSample_IndxInDataSet + aSampleSize - 1;

        if (pLastElementInSample_IndxInDataSet > aDataSet.size() - 1) {
            pLastElementInSample_IndxInDataSet = aDataSet.size() - 1;
        }
    } 
}
//+----------------------------------------------------------------------------------+
void divideSamplesIntoBatches(vector<vector<timeStepStructure>>& aSampleSet, vector<vector<vector<timeStepStructure>>>& aBatchSet, int aBatchSize) {
    
    int pFirstElementInSample_IndxInSampleSet = 0;
    int pLastElementInSample_IndxInSampleSet = aBatchSize - 1;

    while(pLastElementInSample_IndxInSampleSet != aSampleSet.size() - 1){
        vector<vector<timeStepStructure>> pBatch;

        moveElements(aSampleSet, pBatch, pFirstElementInSample_IndxInSampleSet, pLastElementInSample_IndxInSampleSet);
        aBatchSet.push_back(pBatch);

        pFirstElementInSample_IndxInSampleSet = pLastElementInSample_IndxInSampleSet + 1;
        pLastElementInSample_IndxInSampleSet = pFirstElementInSample_IndxInSampleSet + aBatchSize - 1;

        if (pLastElementInSample_IndxInSampleSet > aSampleSet.size() - 1) {
            pLastElementInSample_IndxInSampleSet = aSampleSet.size() - 1;
        }
    }
}
//+----------------------------------------------------------------------------------+
void normalizeData_0_1(vector<timeStepStructure>& aDataSet, vector<timeStepStructure>& aNormalizedSetOfData) {

    auto pMaxValue = max_element(aDataSet.begin(), aDataSet.end(), [](timeStepStructure a, timeStepStructure b) {return a.price < b.price; });
    auto pMinValue = min_element(aDataSet.begin(), aDataSet.end(), [](timeStepStructure a, timeStepStructure b) {return a.price < b.price; });
    for (auto& element : aDataSet) {
        timeStepStructure pNewElement;
        double pNewNormalizedPriceValue = (element.price - pMinValue->price) / (pMaxValue->price - pMinValue->price);
        pNewElement.price = pNewNormalizedPriceValue;
        pNewElement.datetime = element.datetime;
        aNormalizedSetOfData.push_back(pNewElement);
    }
}
//+----------------------------------------------------------------------------------+
void denormalizeData(vector<timeStepStructure>& aOriginalDataSet, vector<timeStepStructure>& aNormalizedSetOfData, vector<timeStepStructure>& aDenormalizedDataSet) {

    auto pMaxValue = max_element(aOriginalDataSet.begin(), aOriginalDataSet.end(), [](timeStepStructure a, timeStepStructure b) {return a.price < b.price; });
    auto pMinValue = min_element(aOriginalDataSet.begin(), aOriginalDataSet.end(), [](timeStepStructure a, timeStepStructure b) {return a.price < b.price; });
    for (auto& element : aNormalizedSetOfData) {
        timeStepStructure pNewElement;
        double pNewDenormalizedPriceValue = element.price * (pMaxValue->price - pMinValue->price) + pMinValue->price;
        pNewElement.price = pNewDenormalizedPriceValue;
        pNewElement.datetime = element.datetime;
        aDenormalizedDataSet.push_back(pNewElement);
    }
}
//+----------------------------------------------------------------------------------+
void prepareDataForTraining(vector<timeStepStructure>& aDataSet, int aSampleSize, int aBatchSize) {

    if (aDataSet.empty()) return;

    normalizeData_0_1(aDataSet, normalizedTrainingDataSet);
    divideDataSetintoSamples(normalizedTrainingDataSet, setOfSamples, aSampleSize);
    divideSamplesIntoBatches(setOfSamples, setOfBatches, aBatchSize);
}
//+----------------------------------------------------------------------------------+
double sigmoid(double x) {

    return 1 / (1 + exp(-x));
}
//+----------------------------------------------------------------------------------+
double forgetGate(double aInputValue, double aInSTMem, double aInLTMem, weights& aWeights, biases& aBias) {

    double pPercentOfLTMemToRemember = sigmoid(aBias.forgetGateBias + (aInSTMem * aWeights.forgetGateSTMemWeight) + (aInputValue * aWeights.forgetGateInputWeight));
    double pNewLongTermMem = aInLTMem * pPercentOfLTMemToRemember;
    return pNewLongTermMem;
}
//+----------------------------------------------------------------------------------+
double inputGate(double aInputValue, double aInSTMem, double aInLTMem, weights& aWeights, biases& aBias) {

    double pPercentOfPotentialLTMemToRemeber = sigmoid(aBias.inputGateBias_1stStage + (aInSTMem * aWeights.inputGateSTMemWeight_1stStage) + (aInputValue * aWeights.inputGateInputWeight_1stStage));
    double pPotentialLTMem = tanh(aBias.inputGateBias_2ndStage + (aInSTMem * aWeights.inputGateSTMemWeight_2ndStage) + (aInputValue * aWeights.inputGateInputWeight_2ndStage));
    double pNewLongTermMem = aInLTMem + (pPercentOfPotentialLTMemToRemeber * pPotentialLTMem);
    return pNewLongTermMem;

}
//+----------------------------------------------------------------------------------+
double outputGate(double aInputValue, double aInSTMem, double aInLTMem, weights& aWeights, biases& aBias) {

    double pPercentOfPotentialSTMemToRemeber = sigmoid(aBias.outputGateBias + (aInSTMem * aWeights.outputGateSTMemWeight) + (aInputValue * aWeights.outputGateInputWeight));
    double pPotentialSTMem = tanh(aInLTMem);
    double pNewShortTermMem = pPercentOfPotentialSTMemToRemeber * pPotentialSTMem;
    return pNewShortTermMem;
}
//+----------------------------------------------------------------------------------+
lstmCellOutput lstmCell(double aInputValue, double aInSTMem, double aInLTMem, weights& aWeights, biases& aBias) {

    lstmCellOutput pOutput;
    double pNewLongTermMem = forgetGate(aInputValue, aInSTMem, aInLTMem, aWeights, aBias);
    pNewLongTermMem = inputGate(aInputValue, aInSTMem, pNewLongTermMem, aWeights, aBias);
    double pNewShortTermMem = outputGate(aInputValue, aInSTMem, pNewLongTermMem, aWeights, aBias);
    pOutput.shortTermMemory = pNewShortTermMem;
    pOutput.longTermMemory = pNewLongTermMem;
    return pOutput;
}
//+----------------------------------------------------------------------------------+
void calculateGradientsForTimeStep(double aInputValue, double aInSTMem, double aInLTMem, double aLoss, double aOutSTMem, double aOutLTMem, weights& aWeights, biases& aBias, weightGradeints& aWeightGradientsForTimeStep, biasGradients& aBiasGradientsForTimeStep) {

    aWeightGradientsForTimeStep.clearGradients();
    aBiasGradientsForTimeStep.clearGradients();

    /*o*/ double pPerOfPotentSTMemToRem_OutGate = sigmoid(aBias.outputGateBias + (aInSTMem * aWeights.outputGateSTMemWeight) + (aInputValue * aWeights.outputGateInputWeight));
    /*1-tanh^2(ct)*/ double pTanhSqOfOutputLTMemMinusOne = 1 - pow(tanh(aOutLTMem), 2);
    /*ct-1*/ aInLTMem;
    /*sigmoid(zf)*/ double pPerOfLTMemToRem_ForgetGate = sigmoid(aBias.forgetGateBias + (aInSTMem * aWeights.forgetGateSTMemWeight) + (aInputValue * aWeights.forgetGateInputWeight));
    /*1 - sigmoid(zf)*/ double pPerOfLTMemToRem_ForgetGateMinusOne = 1 - pPerOfLTMemToRem_ForgetGate;
    /*xt*/ aInputValue;
    /*ht-1*/ aInSTMem;

    aWeightGradientsForTimeStep.forgetGateInputWeightGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * aInLTMem * pPerOfLTMemToRem_ForgetGate * pPerOfLTMemToRem_ForgetGateMinusOne * aInputValue;
    aWeightGradientsForTimeStep.forgetGateSTMemWeightGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * aInLTMem * pPerOfLTMemToRem_ForgetGate * pPerOfLTMemToRem_ForgetGateMinusOne * aInSTMem;
    aBiasGradientsForTimeStep.forgetGateBiasGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * aInLTMem * pPerOfLTMemToRem_ForgetGate * pPerOfLTMemToRem_ForgetGateMinusOne;

    /*g*/ double pPotentialLTMem_InGate = tanh(aBias.inputGateBias_2ndStage + (aInSTMem * aWeights.inputGateSTMemWeight_2ndStage) + (aInputValue * aWeights.inputGateInputWeight_2ndStage));
    /*sigmoid(zi)*/ double pPerOfPotentLTMemToRem_InGate = sigmoid(aBias.inputGateBias_1stStage + (aInSTMem * aWeights.inputGateSTMemWeight_1stStage) + (aInputValue * aWeights.inputGateInputWeight_1stStage));
    /*1 - sigmoid(zi)*/ double pPerOfPotentLTMemToRem_InGateMinusOne = 1 - pPerOfPotentLTMemToRem_InGate;

    aWeightGradientsForTimeStep.inputGateInputWeight_1stStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPotentialLTMem_InGate * pPerOfPotentLTMemToRem_InGate * pPerOfPotentLTMemToRem_InGateMinusOne * aInputValue;
    aWeightGradientsForTimeStep.inputGateSTMemWeight_1stStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPotentialLTMem_InGate * pPerOfPotentLTMemToRem_InGate * pPerOfPotentLTMemToRem_InGateMinusOne * aInSTMem;
    aBiasGradientsForTimeStep.inputGateBias_1stStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPotentialLTMem_InGate * pPerOfPotentLTMemToRem_InGate * pPerOfPotentLTMemToRem_InGateMinusOne;

    /*i == sigmoid(zi)*/
    /*1-tanh^2(zg)*/ double pTanhSqOfPotentLTMemMinusOne_inGate = 1 - pow(tanh(aBias.inputGateBias_2ndStage + (aInSTMem * aWeights.inputGateSTMemWeight_2ndStage) + (aInputValue * aWeights.inputGateInputWeight_2ndStage)), 2);

    aWeightGradientsForTimeStep.inputGateInputWeight_2ndStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPerOfPotentLTMemToRem_InGate * pTanhSqOfPotentLTMemMinusOne_inGate * aInputValue;
    aWeightGradientsForTimeStep.inputGateSTMemWeight_2ndStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPerOfPotentLTMemToRem_InGate * pTanhSqOfPotentLTMemMinusOne_inGate * aInSTMem;
    aBiasGradientsForTimeStep.inputGateBias_2ndStageGrad = aLoss * pPerOfPotentSTMemToRem_OutGate * pTanhSqOfOutputLTMemMinusOne * pPerOfPotentLTMemToRem_InGate * pTanhSqOfPotentLTMemMinusOne_inGate;

    /*tanh(ct)*/ double pTanhOfOutputLTMem = tanh(aOutLTMem);
    /*o == sigmoid(zo)*/
    /*1 - sigmoid(zo)*/ double pPerOfPotentSTMemToRem_OutGateMinusOne = 1 - pPerOfPotentSTMemToRem_OutGate;

    aWeightGradientsForTimeStep.outputGateInputWeightGrad = aLoss * pTanhOfOutputLTMem * pPerOfPotentSTMemToRem_OutGate * pPerOfPotentSTMemToRem_OutGateMinusOne * aInputValue;
    aWeightGradientsForTimeStep.outputGateSTMemWeightGrad = aLoss * pTanhOfOutputLTMem * pPerOfPotentSTMemToRem_OutGate * pPerOfPotentSTMemToRem_OutGateMinusOne * aInSTMem;
    aBiasGradientsForTimeStep.outputGateBiasGrad = aLoss * pTanhOfOutputLTMem * pPerOfPotentSTMemToRem_OutGate * pPerOfPotentSTMemToRem_OutGateMinusOne;
}
//+----------------------------------------------------------------------------------+
void calculateGradientsForSample(weights& aWeights, biases& aBias, vector<timeStepStructure>& aSample, weightGradeints& aWeightGradientsForSample, biasGradients& aBiasGradientsForSample) {

    aWeightGradientsForSample.clearGradients();
    aBiasGradientsForSample.clearGradients();

    weightGradeints pWeightGradientsForTimeStep;
    biasGradients pBiasGradientsForTimeStep;
    double pInSTMem = 0.0, pInLTMem = 0.0;
    double pLoss, pTarget;

    /*for (auto timeStep = aSample.begin(); timeStep < aSample.end() - 1; timeStep++) {
        lstmCellOutput pLstmCellOutput = lstmCell(timeStep->price, pInSTMem, pInLTMem, aWeights, aBias);
        pInSTMem = pLstmCellOutput.shortTermMemory;
        pInLTMem = pLstmCellOutput.longTermMemory;
    }
    auto pTargetIterator = aSample.end() - 1;
    pTarget = pTargetIterator->price;
    pLoss =  -1.0 * (pTarget - pInSTMem);
    pInSTMem = 0.0;
    pInLTMem = 0.0;*/

    for (auto timeStep = aSample.begin(); timeStep < aSample.end() - 1; timeStep++) {
        auto pTargetIterator = next(timeStep);
        pTarget = pTargetIterator->price;
        lstmCellOutput pLstmCellOutput = lstmCell(timeStep->price, pInSTMem, pInLTMem, aWeights, aBias);
        pLoss = -2.0 * (pTarget - pLstmCellOutput.shortTermMemory);
        //lstmCellOutput pLstmCellOutput = lstmCell(timeStep->price, pInSTMem, pInLTMem, aWeights, aBias);
        calculateGradientsForTimeStep(timeStep->price, pInSTMem, pInLTMem, pLoss, pLstmCellOutput.shortTermMemory, pLstmCellOutput.longTermMemory, aWeights, aBias, pWeightGradientsForTimeStep, pBiasGradientsForTimeStep);
        pInSTMem = pLstmCellOutput.shortTermMemory;
        pInLTMem = pLstmCellOutput.longTermMemory;

        aWeightGradientsForSample.forgetGateInputWeightGrad -= pWeightGradientsForTimeStep.forgetGateInputWeightGrad;
        aWeightGradientsForSample.forgetGateSTMemWeightGrad -= pWeightGradientsForTimeStep.forgetGateSTMemWeightGrad;
        aBiasGradientsForSample.forgetGateBiasGrad -= pBiasGradientsForTimeStep.forgetGateBiasGrad;

        aWeightGradientsForSample.inputGateInputWeight_1stStageGrad -= pWeightGradientsForTimeStep.inputGateInputWeight_1stStageGrad;
        aWeightGradientsForSample.inputGateSTMemWeight_1stStageGrad -= pWeightGradientsForTimeStep.inputGateSTMemWeight_1stStageGrad;
        aBiasGradientsForSample.inputGateBias_1stStageGrad -= pBiasGradientsForTimeStep.inputGateBias_1stStageGrad;

        aWeightGradientsForSample.inputGateInputWeight_2ndStageGrad -= pWeightGradientsForTimeStep.inputGateInputWeight_2ndStageGrad;
        aWeightGradientsForSample.inputGateSTMemWeight_2ndStageGrad -= pWeightGradientsForTimeStep.inputGateSTMemWeight_2ndStageGrad;
        aBiasGradientsForSample.inputGateBias_2ndStageGrad -= pBiasGradientsForTimeStep.inputGateBias_2ndStageGrad;

        aWeightGradientsForSample.outputGateInputWeightGrad -= pWeightGradientsForTimeStep.outputGateInputWeightGrad;
        aWeightGradientsForSample.outputGateSTMemWeightGrad -= pWeightGradientsForTimeStep.outputGateSTMemWeightGrad;
        aBiasGradientsForSample.outputGateBiasGrad -= pBiasGradientsForTimeStep.outputGateBiasGrad;
    }
}
//+----------------------------------------------------------------------------------+
void calculateGradientsForBatch(weights& aWeights, biases& aBias, vector<vector<timeStepStructure>>& aBatch, weightGradeints& aWeightGradientsForBatch, biasGradients& aBiasGradientsForBatch) {

    aWeightGradientsForBatch.clearGradients();
    aBiasGradientsForBatch.clearGradients();

    weightGradeints pWeightGradientsForSample;
    biasGradients pBiasGradientsForSample;

    int pBatchSize = aBatch.size();
    for (auto& sample : aBatch) {
        calculateGradientsForSample(aWeights, aBias, sample, pWeightGradientsForSample, pBiasGradientsForSample);

        aWeightGradientsForBatch.forgetGateInputWeightGrad += pWeightGradientsForSample.forgetGateInputWeightGrad;
        aWeightGradientsForBatch.forgetGateSTMemWeightGrad += pWeightGradientsForSample.forgetGateSTMemWeightGrad;
        aBiasGradientsForBatch.forgetGateBiasGrad += pBiasGradientsForSample.forgetGateBiasGrad;

        aWeightGradientsForBatch.inputGateInputWeight_1stStageGrad += pWeightGradientsForSample.inputGateInputWeight_1stStageGrad;
        aWeightGradientsForBatch.inputGateSTMemWeight_1stStageGrad += pWeightGradientsForSample.inputGateSTMemWeight_1stStageGrad;
        aBiasGradientsForBatch.inputGateBias_1stStageGrad += pBiasGradientsForSample.inputGateBias_1stStageGrad;

        aWeightGradientsForBatch.inputGateInputWeight_2ndStageGrad += pWeightGradientsForSample.inputGateInputWeight_2ndStageGrad;
        aWeightGradientsForBatch.inputGateSTMemWeight_2ndStageGrad += pWeightGradientsForSample.inputGateSTMemWeight_2ndStageGrad;
        aBiasGradientsForBatch.inputGateBias_2ndStageGrad += pBiasGradientsForSample.inputGateBias_2ndStageGrad;

        aWeightGradientsForBatch.outputGateInputWeightGrad += pWeightGradientsForSample.outputGateInputWeightGrad;
        aWeightGradientsForBatch.outputGateSTMemWeightGrad += pWeightGradientsForSample.outputGateSTMemWeightGrad;
        aBiasGradientsForBatch.outputGateBiasGrad += pBiasGradientsForSample.outputGateBiasGrad;
    }
    aWeightGradientsForBatch.forgetGateInputWeightGrad /= pBatchSize;
    aWeightGradientsForBatch.forgetGateSTMemWeightGrad /= pBatchSize;
    aBiasGradientsForBatch.forgetGateBiasGrad /= pBatchSize;

    aWeightGradientsForBatch.inputGateInputWeight_1stStageGrad /= pBatchSize;
    aWeightGradientsForBatch.inputGateSTMemWeight_1stStageGrad /= pBatchSize;
    aBiasGradientsForBatch.inputGateBias_1stStageGrad /= pBatchSize;

    aWeightGradientsForBatch.inputGateInputWeight_2ndStageGrad /= pBatchSize;
    aWeightGradientsForBatch.inputGateSTMemWeight_2ndStageGrad /= pBatchSize;
    aBiasGradientsForBatch.inputGateBias_2ndStageGrad /= pBatchSize;

    aWeightGradientsForBatch.outputGateInputWeightGrad /= pBatchSize;
    aWeightGradientsForBatch.outputGateSTMemWeightGrad /= pBatchSize;
    aBiasGradientsForBatch.outputGateBiasGrad /= pBatchSize;
}
//+----------------------------------------------------------------------------------+
void calculateNewWeightsAndBiasesForNextBatch(weights& aWeights, biases& aBias, vector<vector<timeStepStructure>>& aBatch, double aLearningRate) {

    weightGradeints pWeightGradientsForBatch;
    biasGradients pBiasGradientsForBatch;

    calculateGradientsForBatch(aWeights, aBias, aBatch, pWeightGradientsForBatch, pBiasGradientsForBatch);

    //Optimization of weights and biases based on the current batch, these new weights and biases will be input parameters for the next batch

    aWeights.forgetGateInputWeight += (aLearningRate * pWeightGradientsForBatch.forgetGateInputWeightGrad);
    aWeights.forgetGateSTMemWeight += (aLearningRate * pWeightGradientsForBatch.forgetGateSTMemWeightGrad);
    aBias.forgetGateBias += (aLearningRate * pBiasGradientsForBatch.forgetGateBiasGrad);

    aWeights.inputGateInputWeight_1stStage += (aLearningRate * pWeightGradientsForBatch.inputGateInputWeight_1stStageGrad);
    aWeights.inputGateSTMemWeight_1stStage += (aLearningRate * pWeightGradientsForBatch.inputGateSTMemWeight_1stStageGrad);
    aBias.inputGateBias_1stStage += (aLearningRate * pBiasGradientsForBatch.inputGateBias_1stStageGrad);

    aWeights.inputGateInputWeight_2ndStage += (aLearningRate * pWeightGradientsForBatch.inputGateInputWeight_2ndStageGrad);
    aWeights.inputGateSTMemWeight_2ndStage += (aLearningRate * pWeightGradientsForBatch.inputGateSTMemWeight_2ndStageGrad);
    aBias.inputGateBias_2ndStage += (aLearningRate * pBiasGradientsForBatch.inputGateBias_2ndStageGrad);

    aWeights.outputGateInputWeight += (aLearningRate * pWeightGradientsForBatch.outputGateInputWeightGrad);
    aWeights.outputGateSTMemWeight += (aLearningRate * pWeightGradientsForBatch.outputGateSTMemWeightGrad);
    aBias.outputGateBias += (aLearningRate * pBiasGradientsForBatch.outputGateBiasGrad);
}
//+----------------------------------------------------------------------------------+
void header() {

    cout << "                      !!  INFORMATION !!" << endl;
    cout << "       -- Single input and output feature LSTM network --" << endl;
    cout << "       -- Single hidden layer and single stacked layer LSTM --" << endl;
    cout << "       -- Gradient Descent optimizer --" << endl;
}
//+----------------------------------------------------------------------------------+
void trainLstmNetwork(weights& aWeights, biases& aBias, vector<vector<vector<timeStepStructure>>>& aSetOfBatches) {

    fillDataSetVectorFromCsv(trainingDataSet, "AAPL_Train.csv");

    int pSampleSize, pBatchSize, pNumOfEpochs;
    double pLearningRate;

    header();

    do{
        cin.clear();

        cout << "\n\n---------------------------------------------------------" << endl;
        cout << "Make sure, batch size * sample size < training dataset size AND sample size >= 2" << endl;
        cout << "\n\nInput sample size (sequence length + label): ";
        cin >> pSampleSize;
        cout << "\nInput batch size: ";
        cin >> pBatchSize;
        cout << "\nInput learning rate: ";
        cin >> pLearningRate;
        cout << "\nInput number of epochs: ";
        cin >> pNumOfEpochs;
    }while (pSampleSize * pBatchSize > trainingDataSet.size() || pSampleSize < 2);

    prepareDataForTraining(trainingDataSet, (int)pSampleSize, (int)pBatchSize);

    for (int pEpoch = 0; pEpoch < (int)pNumOfEpochs; pEpoch++) {
        for (auto& pBatch : aSetOfBatches) {
            calculateNewWeightsAndBiasesForNextBatch(aWeights, aBias, pBatch, pLearningRate);
        }
    }
}
//+----------------------------------------------------------------------------------+
void runLstmNetwork(weights& aWeights, biases& aBias, vector<timeStepStructure>& aSetOfData) {

    double pInSTMem = 0.0, pInLTMem = 0.0;
    lstmCellOutput pLstmCellOutput;

    for (int i = 0; i < aSetOfData.size(); i++) {
        pLstmCellOutput = lstmCell(aSetOfData[i].price, pInSTMem, pInLTMem, aWeights, aBias);
        
        pInSTMem = pLstmCellOutput.shortTermMemory;
        pInLTMem = pLstmCellOutput.longTermMemory;
        timeStepStructure pNewElement;
        timeStepStructure pNextTimeStep;
        if (i + 1 != aSetOfData.size()) pNextTimeStep = aSetOfData[i + 1];
        pNewElement.price = pInSTMem;
        pNewElement.datetime = pNextTimeStep.datetime;
        normalizedPredictedData.push_back(pNewElement);
    }
}
//+----------------------------------------------------------------------------------+
void runEvaluation(weights& aWeights, biases& aBias, vector<timeStepStructure>& aEvalPriceData) {

    fillDataSetVectorFromCsv(aEvalPriceData, "AAPL_Eval.csv");
    normalizeData_0_1(aEvalPriceData, normalizedEvalDataSet);
    runLstmNetwork(aWeights, aBias, normalizedEvalDataSet);
    denormalizeData(aEvalPriceData, normalizedPredictedData, denormalizedPredictedData);
}
//+----------------------------------------------------------------------------------+
void drawLineCharts(vector<timeStepStructure>& aDenormalizedPredictedPriceData, vector<timeStepStructure>& aActualPriceData) {

    //Please implement your own way of plotting the predictions against the true values
}
//+----------------------------------------------------------------------------------+
void writePredictedPriceDataIntoCsv(vector<timeStepStructure>& aDenormalizedPredictedPriceData, string aFileName) {

    ofstream pCsvFile(aFileName, ios::out | ios::app);
    if (!pCsvFile.is_open()) {
        cerr << "Error opening file!" << std::endl;
        return;
    }
    if (pCsvFile.tellp() == 0) {
        pCsvFile << "Date,Closing Price" << endl;
    }
    for (auto& pTimestep : aDenormalizedPredictedPriceData) {
        pCsvFile << pTimestep.datetime << "," << pTimestep.price << endl;
    }
    pCsvFile.close();
}
//+----------------------------------------------------------------------------------+
int main() {
   
    weights allWeights;
    biases allBiases;

    trainLstmNetwork(allWeights, allBiases, setOfBatches);
    runEvaluation(allWeights, allBiases, evalDataSet);
    writePredictedPriceDataIntoCsv(denormalizedPredictedData, "AAPL_PredictedPrices.csv");
    drawLineCharts(denormalizedPredictedData, evalDataSet);
}