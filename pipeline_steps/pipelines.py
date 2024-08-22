"""
Document Processing Pipeline

Document Classification Pipeline: Classification of PDFs as machine-readable vs Non-Machine readable. Machine Readable PDFs are further classified as genomic reports or others, diagnosis and organisation information is extracted.

Machine-Readable Pipeline: Classified MR documents are used as inputs for end to end pipeline for extracting structured information from PDFs.

"""
import sys
import os
import logging
import glob
import time
import json
import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)

from docprocessing import docprocessor
from dataclasses import dataclass, field

logging.basicConfig(filename=os.path.join(os.getcwd(),"log.txt"),
level=logging.ERROR, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@dataclass
class DataStore:
    """
    Dataclass to store and create directories for the files and intermediate results.
    """

    # Storage base path
    # raw data storage
    STORAGE_PATH: str
    def __post_init__(self):
        self.raw_data: str = os.path.join(self.STORAGE_PATH, "data", "raw_data")

        self.basepath: str = os.path.join(
            self.STORAGE_PATH, r"data", "phase_1_poc_{}".format(time.strftime("%Y%m%d-%H%M%S"))
        )
        # extracted data
        self.extracted_data: str = os.path.join(self.basepath, "extracted_data")
        # MR Docs
        self.mr_docs: str = os.path.join(self.basepath, "mr_data")

        # intermediate_output
        self.intermediate_output: str = os.path.join(self.basepath, "intermediate_output")
        # redcap form output
        self.redcap_output: str = os.path.join(self.basepath, "gold_standard_form_output")


class Pipeline:
    """
    Create a pipeline of steps for end-to-end document processing and information extraction.
    steps: list = [('document_classification', True/False), ('mr_pipeline', True/False)]
    """

    def __init__(
        self, steps: list = [("document_classification", True), ("mr_pipeline", False), ("storage_path", os.getcwd())]
    ):  
        self.STORAGE_PATH = steps[2][1]
        self.datastore = DataStore(STORAGE_PATH=self.STORAGE_PATH)
        self.steps = steps

    def __create_dirs(self):
        """
        Data directory creation prior to starting the pipeline steps.

            Parameters: None

        """
        # create the necessary datastore folders before starting the pipeline steps
        try:
            os.makedirs(self.datastore.basepath)
            os.makedirs(self.datastore.extracted_data)
            os.makedirs(self.datastore.mr_docs)
            os.makedirs(self.datastore.intermediate_output)
            os.makedirs(self.datastore.redcap_output)
            print("Datastore folders created successfully")
        except OSError as error:
            print("folders cannot be created")

    def __check_raw_dir(self):
        """
        Check if raw data has been stored in the raw_data folder

            Parameters: None

        """
        dir = os.listdir(self.datastore.raw_data)
        if len(dir) == 0:
            return False
        else:
            return True

    def __move_files_to_dir(self, src_dir: str, dest_dir: str):
        """
        Move files from temporary folder to another folder.

            Parameters:
                src_dir(str): source directory path
                dest_dir(str): destination directory path

        """
        rawdata = os.listdir(src_dir)
        for f in rawdata:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dest_dir, f)
            shutil.move(src_path, dst_path)

    def __copy_mr_files_from_json(self, results: list, src_dir: str, dest_dir: str):
        """
        Copy MR files from src to destination folder based on JSON result

            Parameters:
                results(list): JSON output containing information on document classification task.
                src_dir(str): source directory path
                dest_dir(str): destination directory path

        """
        for each_file_info in results:
            if (
                each_file_info["is_machine_readable"]
                and each_file_info["is_genomic_report"]
            ):
                src_file = os.path.join(src_dir, each_file_info["filename"])
                dst_file = os.path.join(dest_dir, each_file_info["filename"])
                shutil.copy(src_file, dst_file)

    def _document_classification(self) -> list:
        """
        Classify each document within the extracted_data folder as mr vs non_mr, mr files are further classified
        as genomic reports or not based on diagnosis; diagnosis and organisation information is extracted and returned with
        the output.

            Parameters:
                None

        """

        results = []
        pdf_files = glob.glob("%s/*.pdf" % self.datastore.extracted_data)
        for each_file in pdf_files:
            try:
                tic = time.perf_counter()
                logger.info("Processing file: {}".format(each_file))
                doc = docprocessor.DocClass(each_file)
                results.append(doc.document_classification_per_file())
                toc = time.perf_counter()
                logger.info(f"Time taken to process file: {toc - tic:0.4f} seconds")
            except Exception as e:
                logger.error("Filename: {}".format(each_file))
                logger.error(e, exc_info=True)
        return results

    def __write_result_json(self, results: list, opt: str = "doc_classification"):
        """
        Write outputs of metadata, intermediate and redcap output information obtained.

            Parameters:
                results(list): JSON structure results
                opt(str) == ['doc_classification', 'intermediate_outputs', 'redcap_outputs']
        """
        if opt == "doc_classification":
            with open(
                os.path.join(
                    self.datastore.intermediate_output, "classification_results.json"
                ),
                "w",
            ) as f:
                json.dump(results, f, cls=NpEncoder)
                logger.info(
                    "Classification results are written in intermediate_output\classification_results.json"
                )

    def __check_pipeline_steps(self) -> bool:
        """
        Check whether the steps for the pipeline exists

            Parameters: None
        """
        step_present = False
        for step in self.steps:
            if step[0] in ["document_classification", "mr_docs"]:
                step_present = True
        return step_present

    def __check_pipeline_step_exists(
        self, pipeline_name: str = "document_classification"
    ) -> bool:
        """
        Check whether a single step is present in the pipeline list provided.

            Parameters:
                pipeline_name(str) : name of the pipeline step to be performed. default is 'document_classification'. To perform extraction, use 'mr_pipeline'.

            Returns:
                step_present(bool): returns True if the step is present for execution.
        """
        step_present = False
        for step in self.steps:
            if step[0] == pipeline_name and step[1] == True:
                step_present = True
        return step_present

    def __get_organisation_diagnosis_info(self, f):
        """
        Get the diagnosis and organisation information of a document from the result obtained with the document classification
        pipeline.

            Parameters:
                f(str): path to the file

            Returns:
                diagnosis(str): diagnosis present in the document
                organisation(str): organisation name in the document

        """
        diagnosis = organisation = ""

        def convert_results_dataframe(path):
            """
            Get the JSON file converted into pandas dataframe and retrieve the diagnosis and organisation.
            """
            result_data = pd.read_json(r"{}".format(path))
            filtered_data = result_data.loc[result_data["filename"] == f]
            return (
                filtered_data["diagnosis"].iloc[0],
                filtered_data["organisation"].iloc[0]
            )

        path = Path(
            os.path.join(
                self.datastore.intermediate_output, "classification_results.json"
            )
        )
        if path.is_file():
            diagnosis, organisation = convert_results_dataframe(path)
        else:
            logger.error(
                "document classification result file is not found. Please ensure that the document classification pipeline is run first."
            )

        return diagnosis, organisation

    def __mr_process(self, filename):
        """
        Perform MR steps on a single document.

            Parameters:
                filename(str): Path to the file

        """
        diagnosis, organisation = self.__get_organisation_diagnosis_info(filename)
        
        doc = docprocessor.DocSpec(
            os.path.join(self.datastore.mr_docs, filename), diagnosis, organisation
        )
        extracted_report = doc.extract_report_information()
        json_intermediate_result = extracted_report.to_dict()
        json_redcap_result = extracted_report.final_result_summary
        out_path_intermediate_json = os.path.join(
            self.datastore.intermediate_output,
            "{}_intermediate_output.json".format(filename),
        )
        out_path_redcap_json = os.path.join(
            self.datastore.redcap_output, "{}_gold_standard_form_output.json".format(filename)
        )
        with open(out_path_intermediate_json, "w+") as f:
            json.dump(
                json_intermediate_result,
                f,
                cls=NpEncoder,
                ensure_ascii=True,
                indent=4,
            )

        with open(out_path_redcap_json, "w+") as f:
            json.dump(
                json_redcap_result,
                f,
                cls=NpEncoder,
                ensure_ascii=True,
                indent=4,
            )

    def __start_mr_pipeline(self):
        """
        Perform the steps on MR files

            Parameters: None

        """

        mr_data = os.listdir(self.datastore.mr_docs)
        if len(mr_data) != 0:
            for f in mr_data:

                p = multiprocessing.Process(
                    target=self.__mr_process(f), name="mr_process_per_file"
                )
                p.start()

                # Wait 5 minutes per file
                p.join(300)
                p.terminate()
                p.kill()
                p.join()

                logger.info("MR pipeline completed")

    def start_pipeline(self):
        """
        Based on the input, perform the different pipeline steps.

            Parameters: None

        """
        # perform the different pipeline steps
        if self.__check_pipeline_steps:
            logger.info("Pipeline steps are provided: {}".format(self.steps))
            if self.__check_pipeline_step_exists(
                pipeline_name="document_classification"
            ):
                # check if raw data is provided in the raw_data folder
                if self.__check_raw_dir():
                    self.__create_dirs()
                    # move raw data to extracted data folder for further pipeline usage
                    self.__move_files_to_dir(
                        self.datastore.raw_data, self.datastore.extracted_data
                    )
                    # documents in extracted data folder are classified
                    results = self._document_classification()
                    # classification metainformation results are output in JSON and stored in intermediate results folder
                    self.__write_result_json(results, opt="doc_classification")
                    # MR files from classification results are copied from extracted_data and stored in mr_docs
                    self.__copy_mr_files_from_json(
                        results, self.datastore.extracted_data, self.datastore.mr_docs
                    )
                    logger.info("MR files are stored in mr_docs folder")
                else:
                    logging.exception(
                        "No files present in the directory. Please add the raw data in raw_data folder."
                    )

            if self.__check_pipeline_step_exists(pipeline_name="mr_pipeline"):
                self.__start_mr_pipeline()

        else:
            logger.error("pipeline steps are not provided")
