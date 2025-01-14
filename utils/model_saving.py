# utils/model_saving.py
import tensorflow as tf

def save_model(model, file_path):
    model.save(file_path)