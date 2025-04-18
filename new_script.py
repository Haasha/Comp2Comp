'''
This script takes a CT Scan direcotry as argument and does the following:
Load the files
Performe spine segmentation and dump those nifti segmentaiton files
Perform Muscle adipose tissue and dump those nifti segmentaiton files
Perform liver spleen pancreas segmentation and dump those nifti segmentaiton files
'''
#  the user passes --input_path on the command line
# Example of running the script:
# python new_script.py --i some_path
#!/usr/bin/env python
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from comp2comp.inference_pipeline import InferencePipeline
from comp2comp.io import io
from comp2comp.liver_spleen_pancreas import liver_spleen_pancreas
from comp2comp.muscle_adipose_tissue import muscle_adipose_tissue
from comp2comp.spine import spine
from comp2comp.utils.process import process_3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def SpinePipelineBuilder(path, args):
    pipeline = InferencePipeline(
        [
            io.DicomToNifti(path),
            spine.SpineSegmentation(args.spine_model, save=True),
        ]
    )
    return pipeline

def MuscleAdiposeWrapper(path, args):
    pipeline = InferencePipeline(
        [ 
            MuscleAdiposeTissuePipelineBuilder(path,args) 
        ]
    )
    return pipeline

def MuscleAdiposeTissuePipelineBuilder(path, args):
    pipeline = InferencePipeline(
        [
            io.DicomToNifti(path),
            muscle_adipose_tissue.MuscleAdiposeTissueSegmentation(
                16, args.muscle_fat_model
            ),
            muscle_adipose_tissue.MuscleAdiposeTissuePostProcessing()
        ]
    )
    return pipeline


def LiverSpleenPancreasPipelineBuilder(path, args):
    pipeline = InferencePipeline(
        [
            io.DicomToNifti(path),
            liver_spleen_pancreas.LiverSpleenPancreasSegmentation()
        ]
    )
    return pipeline

def argument_parser(extra):
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--input_path", "-i", type=str, required=True)
    base_parser.add_argument("--output_path", "-o", type=str)
    base_parser.add_argument("--save_segmentations", action="store_true")
    base_parser.add_argument("--overwrite_outputs", action="store_true")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="pipeline", help="Pipeline to run", required=False)

    # Add the help option to each subparser
    muscle_adipose_tissue_parser = subparsers.add_parser(
        "muscle_adipose_tissue", parents=[base_parser]
    )

    muscle_adipose_tissue_parser.add_argument(
        "--muscle_fat_model", default="stanford_v0.0.2", type=str
    )
    
    # Spine
    spine_parser = subparsers.add_parser("spine", parents=[base_parser])
    spine_parser.add_argument("--spine_model", default="ts_spine", type=str)

    # Spine + muscle + fat
    spine_muscle_adipose_tissue_parser = subparsers.add_parser(
        "spine_muscle_adipose_tissue", parents=[base_parser]
    )
    spine_muscle_adipose_tissue_parser.add_argument(
        "--muscle_fat_model", default="stanford_v0.0.2", type=str
    )
    spine_muscle_adipose_tissue_parser.add_argument(
        "--spine_model", default="ts_spine", type=str
    )

    # Liver spleen pancreas
    liver_spleen_pancreas = subparsers.add_parser(
        "liver_spleen_pancreas", parents=[base_parser]
    )

    # Aortic calcium
    aortic_calcium = subparsers.add_parser(
        "aortic_calcium", parents=[base_parser])
    
    aortic_calcium.add_argument(
        "--threshold", default="adaptive", type=str
    )
    
    # Contrast phase
    contrast_phase_parser = subparsers.add_parser(
        "contrast_phase", parents=[base_parser]
    )

    hip_parser = subparsers.add_parser("hip", parents=[base_parser])
    hip_parser.add_argument(
        "--hip_model",
        default="ts_hip",
        type=str,
    )

    # AAA
    aorta_diameter_parser = subparsers.add_parser("aaa", help="aorta diameter", parents=[base_parser])

    aorta_diameter_parser.add_argument(
        "--aorta_model",
        default="ts_spine",
        type=str,
        help="aorta model to use for inference",
    )

    aorta_diameter_parser.add_argument(
        "--spine_model",
        default="ts_spine",
        type=str,
        help="spine model to use for inference",
    )

    all_parser = subparsers.add_parser("all", parents=[base_parser])
    all_parser.add_argument(
        "--muscle_fat_model",
        default="stanford_v0.0.2",
        type=str,
    )
    all_parser.add_argument(
        "--spine_model",
        default="ts_spine",
        type=str,
    )
    all_parser.add_argument(
        "--hip_model",
        default="ts_hip",
        type=str,
    )
    return parser




def run_pipeline(pipeline_name, builder_class, extra_args=None):
    parser = argument_parser(extra_args[1])
    # Parse the pipeline argument
    cli_args = [pipeline_name]
    if extra_args:
        cli_args += extra_args
    args = parser.parse_args(cli_args)
    process_3d(args, builder_class)

if __name__ == "__main__":
    import sys

    # the user passes --input_path on the command line
    # Example of running the script:
    # python new_script.py --i some_path

    # Pass the pipeline name explicitly
    run_pipeline("spine", SpinePipelineBuilder, extra_args=sys.argv[1:])
    run_pipeline("muscle_adipose_tissue", MuscleAdiposeWrapper, extra_args=sys.argv[1:])
    run_pipeline("liver_spleen_pancreas", LiverSpleenPancreasPipelineBuilder, extra_args=sys.argv[1:])
