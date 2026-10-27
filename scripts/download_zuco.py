"""Plan or acquire official normal-reading exports with checksums."""
import sys
from eeg_decoding.cli import main
if __name__ == '__main__':
    main(['download-zuco', *sys.argv[1:]])
