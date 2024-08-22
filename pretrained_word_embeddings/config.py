#config file when using local files for pretrained models.
#store them in the folder as shown below
import os
from pathlib import Path

LOCAL_FILES = True

DATAPATH = os.path.join(Path().resolve().parent, "nlp_models")

# while computing the class vector, is word ordering taken into consideration
WORD_ORDERING = False

# pretrained embedding vectors word2vec format or not
W2VFORMAT = True

MODEL_CLASSIFICATION = r"C:\Users\rajenp\Desktop\projects\genomics_workstream\open-publication-repo\genomics_nlp_pipeline\data\nlp_models\biobert-nli"
MODEL_THRESHOLD_CLASSIFICATION = 0.50

MODEL_CONCLUSION =  r"C:\Users\rajenp\Desktop\projects\genomics_workstream\open-publication-repo\genomics_nlp_pipeline\data\nlp_models\multi-qa-MiniLM-L6-cos-v1"
MODEL_THRESHOLD_CONCLUSION = 0.50
