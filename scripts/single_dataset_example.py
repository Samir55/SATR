import os
import argparse
from helper_functions import segment

if __name__ == "__main__":
    # Get the config file path from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", help="path to the yaml config file.", required=True)
    parser.add_argument("-mesh_name", help="name of the mesh to run on", required=True)
    parser.add_argument("-output_dir", help="path to the output dir.", required=True)
    parser.add_argument("--gpu", help="gpu id", default=0, type=int)
    # TODO @ahmed use the gpu option and propagate it to others in the library
    args = parser.parse_args()

    # Read the config file
    assert os.path.isfile(args.cfg)

    segment(
        config_path=args.cfg,
        mesh_name=args.mesh_name,
        output_dir=args.output_dir,
    )
