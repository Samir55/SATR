import argparse
import warnings
warnings.filterwarnings("ignore")
from meshseg.evaluation.eval_functions import *

if __name__ == "__main__":
    # Get the config file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", help="path to the output dir.", required=True)
    parser.add_argument("--fine_grained", help="fine grained evaluation", action="store_true")
    args = parser.parse_args()

    evaluate_faust(args.output_dir, args.fine_grained)