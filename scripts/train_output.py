#!/usr/bin/env python3
"""
Train output gaze integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_neurogate import main as train_main

if __name__ == "__main__":
    train_main(integration_type='output', output_suffix='output')