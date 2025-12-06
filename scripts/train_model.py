import os
from pathlib import Path
import sys
from datetime import datetime

THIS_FILE = Path(__file__).resolve()

# Project root = parent of "scripts"
PROJECT_ROOT = THIS_FILE.parents[1]

# src directory
SRC_DIR = PROJECT_ROOT / "src"

# Put src on sys.path so "neural_decoder" becomes importable
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
    
# ---

"""
Steps to train a new model:
1. Create a new model in model.py
2. Rename ModelName in this file
3. In neural_decoder_trainer.py, make sure that its imported from .model correctly
4. In neural_decoder_trainer.py, check instantiation from the trainModel function.
5. Run the script
6. Collect useful plots and metrics, document everything interesting about the model and put it in your report
7. Rinse and repeat
"""

modelName = 'optimizedBaseline'

# Append timestamp (day_hour_min) to the model name so saved output
# directories look like: speechBaseline4_04_18_03
_timestamp = datetime.now().strftime("%d_%H_%M")
modelName = f"{modelName}_{_timestamp}"
args = {}
# check if we are in the scripts directory, if so, go up one level
if os.getcwd().endswith('scripts'):
	args['datasetPath'] = '../competitionData/allDatasetsLogged.pkl'
	args['outputDir'] = '../speechLogs/' + modelName
else:
	args['datasetPath'] = './competitionData/allDatasetsLogged.pkl'
	args['outputDir'] = './speechLogs/' + modelName

# Optimized
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32
args['lrStart'] = 0.04
args['lrEnd'] = 0.015
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.3
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.3
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

# Unidirectional
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 32
# args['lrStart'] = 0.04
# args['lrEnd'] = 0.015
# args['nUnits'] = 512
# args['nBatch'] = 10000 #3000
# args['nLayers'] = 3
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['dropout'] = 0.3
# args['whiteNoiseSD'] = 1
# args['constantOffsetSD'] = 0.3
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['l2_decay'] = 1e-5

# Transformer
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 32
# args['lrStart'] = 0.05
# args['lrEnd'] = 0.02
# args['nUnits'] = 1024
# args['nHeads'] = 4
# args['nBatch'] = 10000 # 3000
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['dropout'] = 0.1
# args['whiteNoiseSD'] = 0.9
# args['constantOffsetSD'] = 0.3
# args['gaussianSmoothWidth'] = 1.5
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = False
# args['l2_decay'] = 1e-5
# args['cr_lambda'] = 0.1
# # args["transitionLossWeight"] = 0.4


# Ensemble
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 32
# args['lrStart'] = 0.04
# args['lrEnd'] = 0.015
# args['nUnits1'] = 1024
# args['nUnits2'] = 512
# args['nBatch'] = 10000 #3000
# args['nLayers1'] = 5
# args['nLayers2'] = 3
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['dropout'] = 0.3
# args['whiteNoiseSD'] = 0.9
# args['constantOffsetSD'] = 0.3
# args['gaussianSmoothWidth1'] = 1.0
# args['gaussianSmoothWidth2'] = 3.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = True
# args['l2_decay'] = 1e-5
# args['ensembleWeightStart'] =  2.0
# args['ensembleWeightEnd'] = 0.5






# Original Arguments
# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 32
# args['lrStart'] = 0.02
# args['lrEnd'] = 0.02
# args['nUnits'] = 1024
# args['nBatch'] = 10000 #3000
# args['nLayers'] = 5
# args['seed'] = 0
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['dropout'] = 0.4
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 2.0
# args['strideLen'] = 4
# args['kernelLen'] = 32
# args['bidirectional'] = True
# args['l2_decay'] = 1e-5






from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
