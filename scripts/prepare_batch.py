"""Shared-code batch entry point for Windows, Linux and Colab."""
import sys
from eeg_decoding.cli import main
if __name__ == '__main__':
    main(['prepare-batch', *sys.argv[1:]])
