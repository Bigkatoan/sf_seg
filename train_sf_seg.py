#!/usr/bin/env python3
"""Wrapper for src.training.trainer — see that module for details."""
from src.training.trainer import parse_args, merge_config, train

if __name__ == "__main__":
    args = parse_args()
    merge_config(args)
    train(args)
