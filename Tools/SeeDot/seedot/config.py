# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Target word length. Currently set to match the word length of Arduino (2 bytes)
wordLength = 16
availableBitwidths = [8, 16, 32]

# Range of max scale factor used for exploration
maxScaleRange = 0, -wordLength

# tanh approximation limit
tanhLimit = 1.0

# MSBuild location
# Edit the path if not present at the following location
msbuildPathOptions = [r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
                      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
                      ]

# Currently not supported (ddsEnabled = False and vbwEnabled = True) 
ddsEnabled = True
vbwEnabled = True
functionReducedProfiling = True

trimHighestDecile = False

higherOffsetBias = True

fixedPointVbwIteration = False

class Metric:
    accuracy = "acc"
    disagreements = "disagree"
    reducedDisagreements = "red_disagree"
    regressionLoss = "reg_loss"
    default = [accuracy]
    all = [accuracy, disagreements, reducedDisagreements, regressionLoss]

class Algo:
    bonsai = "bonsai"
    lenet = "lenet"
    protonn = "protonn"
    rnn = "rnn"
    rnnpool = "rnnpool"
    mbconv = "mbconv"
    resnet18 = "resnet-18"
    resnet50 = "resnet-50"
    test = "test"
    default = [bonsai, protonn]
    all = [bonsai, lenet, protonn, rnn, rnnpool, mbconv, resnet18, resnet50, test]


class Version:
    fixed = "fixed"
    floatt = "float"
    default = [fixed, floatt]
    all = default


class DatasetType:
    training = "training"
    testing = "testing"
    default = testing
    all = [training, testing]

class ProblemType:
    classification = "classification"
    regression = "regression"
    default = classification
    all = [classification, regression]

class Target:
    arduino = "arduino"
    x86 = "x86"
    default = x86
    all = [arduino, x86]

class Source:
    seedot = "seedot"
    tf = "tf"
    onnx = "onnx"
    default = seedot
    all = [seedot, tf, onnx]