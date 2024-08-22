#config file when using local files for pretrained models.
#store them in the folder as shown below
import os
from pathlib import Path

LOCAL_FILES = False

DATAPATH = os.path.join(Path().resolve().parent, "nlp_models")

# while computing the class vector, is word ordering taken into consideration
WORD_ORDERING = False

# pretrained embedding vectors word2vec format or not
W2VFORMAT = True

MODEL_CLASSIFICATION = os.path.join(DATAPATH, "biobert-nli")
MODEL_THRESHOLD_CLASSIFICATION = 0.50

MODEL_CONCLUSION = os.path.join(DATAPATH, "multi-qa-MiniLM-L6-cos-v1")
MODEL_THRESHOLD_CONCLUSION = 0.50
