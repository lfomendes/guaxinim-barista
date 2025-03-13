"""
Main entry point for the Guaxinim application.
"""
from guaxinim.ui.home import main
import torch
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

if __name__ == "__main__":
    main()
