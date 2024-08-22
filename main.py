# Main File to run the doc processing steps

import sys
import os
import logging
import glob
import time
import json
import argparse
import shutil

from docprocessing import docprocessor
from pipeline_steps.pipelines import Pipeline
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def start_process():
    parser = argparse.ArgumentParser(description="Provide options for pipeline steps")
    parser.add_argument(
        "--pipeline_steps",
        type=int,
        help="1 : document classification only \n 2. MR documents pipeline only \n 3. Both (1) and (2)",
        required=True,
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        help="please provide the path to store the documents and results"
    )
    args = parser.parse_args()
    print(args.storage_path)
    # performs document classification only
    if args.pipeline_steps == 1:
        pipeline = Pipeline(
            steps=[("document_classification", True), ("mr_pipeline", False), ("storage_path", args.storage_path)]
        )
        pipeline.start_pipeline()
    # performs mr pipeline only
    elif args.pipeline_steps == 2:
        pipeline = Pipeline(
            steps=[("document_classification", False), ("mr_pipeline", True), ("storage_path", args.storage_path)]
        )
        pipeline.start_pipeline()
    # performs both document classification and mr pipeline
    elif args.pipeline_steps == 3:
        pipeline = Pipeline(
            steps=[("document_classification", True), ("mr_pipeline", True), ("storage_path", args.storage_path)]
        )
        pipeline.start_pipeline()
    else:
        print("Provide a proper input")
