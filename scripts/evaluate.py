"""Thin entry point; implementation lives in eeg_decoding."""
import sys
from eeg_decoding.cli import main
if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
