import torch
import os

class AbstractTrainer:
    def __init__(self):
        self.model = None
    
    @classmethod
    def train(self):
        return NotImplemented

    def get_model(self):
        return NotImplemented

    def get_model_size(self):
        if self.model is None:
            print("Model is not initialized.")
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_model(self):
        return NotImplemented

    def generate_image(self):
        return NotImplemented