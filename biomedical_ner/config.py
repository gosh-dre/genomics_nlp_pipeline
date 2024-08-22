import os
#download and add the model here 
model_path = os.path.join(os.getcwd(),"data","nlp_models","hunflair", "hunflair-gene-full-v1.0.pt")

NER_OUTPUT_EXPRESSION = r'("[a-zA-Z0-9, ]*"/[a-zA-Z0-9\-/ :]*)'
