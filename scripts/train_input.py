#!/usr/bin/env python3
"""
Train input gaze integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_neurogate import main as train_main

if __name__ == "__main__":
    train_main(integration_type='input', output_suffix='input')