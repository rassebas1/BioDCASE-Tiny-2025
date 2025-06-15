from embedded_code_generation import run_embedded_code_generation
from feature_extraction import run_feature_extraction
from data_preprocessing import run_preprocessing
from config import load_config
from model_training import run_model_training
from unlabeled_evaluator import evaluate_model_from_pipeline_paths
import numpy as np
import serial.tools.list_ports

if __name__ == '__main__':
    config = load_config()
    run_preprocessing(config,augment_training=True)
    run_feature_extraction(config)
    run_model_training(config)
    
    #run_embedded_code_generation(config)
    results = evaluate_model_from_pipeline_paths(config)
    #print("Evaluation results:", results.keys())
    #print("Evaluation summary:", results.summary)
    #outputs= np.load("D:\\Salle\\TFM\\BioDcase25\\BioDCASE-Tiny-2025\\data\\05_reporting\\all_model_outputs.npz")
    #print("Model outputs:", outputs)