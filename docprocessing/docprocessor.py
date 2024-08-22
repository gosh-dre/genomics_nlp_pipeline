"""
Document Processing for Machine-Readable PDFs

Each PDF document is checked whether it is machine readable or not. Meta information extracted: number of pages,
number of tables present, raw text content etc.

The extracted tables are then identified based on the header information and classified as - patient information,
report and authorisation, variant information and, panel and coverage information.

Raw text content is further processed to obtain potential section titles and the content is partitioned into different sections based on corresponding titles.

Information and key-value pairs are extracted from both tabular and section content.
"""
import pdfplumber as pb
import re
import pandas as pd
import numpy as np
import os

from docprocessing.config import (
    TableExtractionConfig,
    SectionExtractionConfig,
    KeyValuePairsConfig,
    OntologyMatcher,
    DocClassConfig,
    OrganisationTemplateConfig,
)
from docprocessing.utils import *

from biomedical_ner import ner

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


class PostProcessReportStructure:
    """
    This class is used for post processing the dataclass variables. All the variables are predefined and specific to the extracted information.
    """

    def __init__(self):
        pass

    def patient_entities(self, patient_info_entities):
        """
        post process patient information
        """
        if "Patient Name" in patient_info_entities.keys():
            if "," in patient_info_entities["Patient Name"]:
                patient_info_entities["f_name"] = patient_info_entities[
                    "Patient Name"
                ].split(",")[1]
                patient_info_entities["l_name"] = patient_info_entities[
                    "Patient Name"
                ].split(",")[0]
            else:
                if patient_info_entities["Patient Name"] != "":
                    patient_info_entities["f_name"] = patient_info_entities[
                        "Patient Name"
                    ].split()[0]
                    patient_info_entities["l_name"] = " ".join(
                        patient_info_entities["Patient Name"].split()[1:]
                    )

        if (
            "Surname" in patient_info_entities.keys()
            and "Forename" in patient_info_entities.keys()
        ):
            patient_info_entities["l_name"] = patient_info_entities["Surname"]
            patient_info_entities["f_name"] = patient_info_entities["Forename"]

        if "Date of Birth" in patient_info_entities.keys():
            date = find_dates(patient_info_entities["Date of Birth"])

        if "MRN number" in patient_info_entities.keys():
            patient_info_entities["gosh_mrn"] = remove_white_spaces(
                patient_info_entities["MRN number"]
            )
        if "NHS number" in patient_info_entities.keys():
            patient_info_entities["nhs_number"] = remove_white_spaces(
                patient_info_entities["NHS number"]
            )
        if "Family number" in patient_info_entities.keys():
            patient_info_entities["f_number"] = remove_white_spaces(
                patient_info_entities["Family number"].strip()
            )
        if "Gender" in patient_info_entities.keys():
            patient_info_entities["gender_sex"] = find_gender(
                patient_info_entities["Gender"]
            )

        return patient_info_entities

    def get_referral_reason(self, section):
        """
        Extract referral reason from section information.
        """
        if "referral reason" in section.keys():
            return section["referral reason"]
        return ""

    def get_diagnostic_conclude(self, section):
        """
        Extract conclusion from section information.
        """
        for key, value in section.items():
            if "diagnostic_conclude" in key.lower():
                return value
        return ""

    def get_classification_conclusion_from_tables(
        self, organisation, key, variant_tables, ontology_variant_classification
    ):
        """
        Get the classification information from the variant tables.
        """
        gene_variant_classification = gene_variant_conclusion = ""
        if "classification" in key.lower() and "classification" in [
            key.lower() for key in variant_tables.keys()
        ]:

            if variant_tables["classification"] is not None:
                    var_class = ""
                    if "\n" in variant_tables["classification"].strip():
                        var_class = replace_delimiter(variant_tables["classification"].strip(),
                        replacewith=" ",
                    )

                    else:
                        var_class = replace_delimiter(variant_tables["classification"].strip())

                    if var_class in ontology_variant_classification.keys():
                        gene_variant_classification = ontology_variant_classification[var_class]

        if "conclusion" in key.lower() and "conclusion" in variant_tables.keys():
            gene_variant_conclusion = variant_tables["conclusion"]
            if gene_variant_classification == "":
                gene_variant_classification = get_classification_from_conc(
                    variant_tables["conclusion"]
                )

        return gene_variant_classification, gene_variant_conclusion

    def get_hgvs_information(self, variant_tables, tagger, gene_name):
        """
        Extract granular information from hgvs variant information.
        """
        hgvs_desc = dna_change_id = amino_change_id = transcript_ref_id = ""
        for key in variant_tables.keys():
            if key is not None and "hgvs" in key and variant_tables[key] is not None:
                if tagger.print_entities_per_sent(variant_tables[key]).keys():
                    if gene_name == "":
                        gene_name = list(
                            tagger.print_entities_per_sent(variant_tables["hgvs"]).keys()
                        )[0]
                chk_nm = []
                hgvs_desc = variant_tables["hgvs"]
                if hgvs_desc:
                    if "c." in hgvs_desc:
                        temp_value = [item for item in hgvs_desc.split() if "c." in item][0]
                        if "NM_" in temp_value:
                            chk_nm = [temp_value.split("c.")[0]]

                        else:
                            chk_nm = [item for item in hgvs_desc.split() if "NM_" in item]
                        dna_change_id = "c." + temp_value.split("c.")[1]
                    if "p." in hgvs_desc:
                        amino_change_id = [
                            item for item in hgvs_desc.split() if "p." in item
                        ][0]

                if chk_nm:
                    remove_non_decimal = re.compile(r"[^\d.]+")
                    transcript = "NM_{}".format(
                        remove_non_decimal.sub("", chk_nm[0].split("NM_")[1])
                    )
                    transcript_ref_id = transcript
            if key is not None and "cdna" in key:
                    dna_change_id = variant_tables[key]
            if key is not None and "protein" in key:
                    amino_change_id = variant_tables[key]
            if gene_name is not None and "NM_" in gene_name:
                temp_name = gene_name
                gene_name = gene_name.split("NM")[0].strip()
                gene_name = re.sub(r'\W+', '', gene_name)
                remove_non_decimal = re.compile(r"[^\d.]+")
                transcript_ref_id = "NM_{}".format(
        remove_non_decimal.sub("", temp_name.split("NM_")[1])
                    )

        return gene_name, hgvs_desc, dna_change_id, amino_change_id, transcript_ref_id

    def get_inheritance(self, variant_tables, ontology_inheritance_loinc):
        """
        Extract inheritance information
        """
        inheritance = ""
        if "inheritance" in variant_tables.keys():
            if variant_tables["inheritance"] in ontology_inheritance_loinc.keys():
                inheritance = ontology_inheritance_loinc[variant_tables["inheritance"]]

        return inheritance

    def get_zygosity(self, variant_tables, ontology_zygosity):
        """
        Extract zygosity information
        """
        zygosity = ""
        if "zygosity" in variant_tables.keys():
            if variant_tables["zygosity"] in ontology_zygosity.keys():
                zygosity = ontology_zygosity[variant_tables["zygosity"]]
        return zygosity


@dataclass_json
@dataclass
class ReportStructure(PostProcessReportStructure):
    """
    DataClass to store the result for each document.
    """

    # meta information
    doc_path: str
    diagnosis: str
    organisation: str
    number_of_pages: int
    number_of_tables: int
    text_present: bool
    tables_present: bool
    is_mr: bool
    raw_text: dict = field(default_factory=lambda: {})
    section: dict = field(default_factory=lambda: {})
    processed_tables: dict = field(default_factory=lambda: {})

    report_structure_info: dict = field(default_factory=lambda: {})

    variant_info_entities: list = field(default_factory=lambda: [])
    coverage_and_panel_entities: list = field(default_factory=lambda: [])
    patient_info_entities: list = field(default_factory=lambda: [])
    patient_info_redcap_entities: list = field(default_factory=lambda: [])

    section_detected_genes: dict = field(default_factory=lambda: {})

    gene_list_from_screening: list = field(default_factory=lambda: [])

    report_conclusion: str = ""
    gene_variant_conclusion: str = ""
    gene_variant_classification: str = ""
    referral_reason: str = ""
    diagnostic_conclude: str = ""
    overall_inter: str = ""
    pos_neg_result: int = 3
    recd_recommen: str = ""
    gene_name: str = ""
    hgvs_desc: str = ""
    transcript_ref_id: str = ""
    dna_change_id: str = ""
    amino_change_id: str = ""
    inheritance: str = ""
    zygosity: str = ""
    pland_testmethod: str = ""
    panelapp_url: list = field(default_factory=lambda: [])
    gene_list: list = field(default_factory=lambda: [])
    variants_identified: list = field(default_factory=lambda: [])

    tagger = ner.NER()
    ontology = OntologyMatcher()
    postprocessor = PostProcessReportStructure()

    def __post_init__(self):
        # get the meta information about the report
        self.report_structure_info = {
            "number of pages": self.number_of_pages,
            "number of tables": self.number_of_tables,
            "is text present": self.text_present,
            "is tables present": self.tables_present,
        }

        # extract patient information based on gold standard annotation
        self.patient_info_entities = self.postprocessor.patient_entities(
            self.patient_info_entities
        )

        # extract referral reason information based on gold standard annotation
        self.referral_reason = self.postprocessor.get_referral_reason(self.section)

        # extract report conclusion information
        self.report_conclusion = self.postprocessor.get_diagnostic_conclude(
            self.section
        )

        self.pos_neg_result = check_pos_neg_variants_from_summary(self.diagnostic_conclude)

        if self.pos_neg_result:
            # extract entities and postprocess from variant info key value pairs
            for num, variant_tables in enumerate(self.variant_info_entities):
                self.gene_variant_classification = (
                    self.gene_name
                ) = (
                    self.hgvs_desc
                ) = (
                    self.dna_change_id
                ) = (
                    self.amino_change_id
                ) = (
                    self.transcript_ref_id
                ) = self.inheritance = self.zygosity = evidence = ""

                if "gene" in variant_tables.keys():
                    self.gene_name = variant_tables["gene"]

                for key, value in variant_tables.items():
                    if key is not None:
                        

                        (
                            self.gene_variant_classification,
                            self.gene_variant_conclusion,
                        ) = self.postprocessor.get_classification_conclusion_from_tables(
                            self.organisation,
                            key,
                            variant_tables,
                            self.ontology.variant_classification,
                        )


                # extract hgvs desc based on gold standard annotation
                (
                    self.gene_name,
                    self.hgvs_desc,
                    self.dna_change_id,
                    self.amino_change_id,
                    self.transcript_ref_id,
                ) = self.postprocessor.get_hgvs_information(
                    variant_tables, self.tagger, self.gene_name
                )

                # extract inheritance
                self.inheritance = self.postprocessor.get_inheritance(
                    variant_tables, self.ontology.inheritance_loinc
                )

                # extract zygosity
                self.zygosity = self.postprocessor.get_zygosity(
                    variant_tables, self.ontology.zygosity
                )
               
                # append variants identified from tabular information
                if self.gene_name:
                    self.variants_identified.append(
                        {
                            "gene_name": self.gene_name,
                            "transcript_ref_id": self.transcript_ref_id,
                            "dna_change_id": self.dna_change_id,
                            "amino_change_id": self.amino_change_id,
                            "inheritance": self.inheritance,
                            "zygosity": self.zygosity,
                            "classification": self.gene_variant_classification,
                            "gene_information": "",
                            "variant_evidence": evidence,
                        }
                    )

            # extract diagnostic conclusion based on gold standard annotation
            if self.report_conclusion:
                self.diagnostic_conclude = self.report_conclusion
            if self.diagnostic_conclude == "":
                if self.gene_variant_conclusion != "":
                    self.diagnostic_conclude = self.gene_variant_conclusion

            # extract overall interpretation information based on gold standard annotation
            if "overall_inter" in self.section.keys():
                self.overall_inter = self.section["overall_inter"]

            # extract recommendation information based on gold standard annotation
            if "recd_recommen" in self.section.keys():
                self.recd_recommen = self.section["recd_recommen"]
            #extract test method description based on gold standard annotation
            if "plandtestmethod" in self.section.keys():
                self.pland_testmethod = self.section["plandtestmethod"]

            # extract gene_panel_list 
            if "panel apps and versions" in self.coverage_and_panel_entities.keys():
                self.gene_list = self.coverage_and_panel_entities["panel apps and versions"]
                if "gene list" in self.coverage_and_panel_entities.keys():
                    self.panelapp_url = self.coverage_and_panel_entities["gene list"]
            else:
                if "gene list" in self.coverage_and_panel_entities.keys():
                    self.gene_list = self.coverage_and_panel_entities["gene list"]


            self.section_detected_genes = {
                key: self.tagger.print_entities_per_sent(value.strip())
                for key, value in self.section.items()
            }

            # cystic fibrosis reports do not contain the gene list as gene names but as amino and dna ids
            if self.diagnosis == "cystic_fibrosis":
                self.gene_list_from_screening = ["CFTR"]
                if "plandtestmethod" in self.section.keys():
                    self.gene_list = [
                        word
                        for word in self.section["plandtestmethod"].split() 
                        if "c." in word or "p." in word
                    ]

                    if len(self.gene_list) == 0:
                        self.gene_list = ["CFTR"]

        

            if len(self.gene_list_from_screening) == 0:
                    self.gene_list_from_screening = flatten_lists(
                        [
                            [k for k, v in value.items() if v == "Gene"]
                            for key, value in self.section_detected_genes.items()
                            if key in ["plandtestmethod"]
                        ]
                    )

            if len(self.gene_list) == 0:
                    if "plandtestmethod" in self.section.keys():
                        self.gene_list = get_panel_list_regex_pattern(
                            self.section["plandtestmethod"]
                        )
                    

            # if panels are not present check if it exists as a gene screening list

            if len(self.gene_list) == 0:
                    if self.gene_list_from_screening:
                        self.gene_list = self.gene_list_from_screening
        
            # append variants identified from text
            self.variants_identified.extend(
                    get_variants(
                        "\n".join(self.raw_text.values()),
                        self.diagnosis,
                        self.organisation,
                        self.gene_list_from_screening,
                        self.pland_testmethod,
                        self.overall_inter,
                        self.tagger,
                    )
                )
            

            # if no information found in test information, append the identified variants
            if len(self.gene_list) == 0:
                self.gene_list = [
                    value
                    for keypairs in self.variants_identified
                    for key, value in keypairs.items()
                    if key == "gene_name"
                ]
            # extract transcript IDs of the variants
            
            self.variants_identified = get_transcripts_id(
                    self.variants_identified, self.pland_testmethod
                )


        self.final_result_summary = {
            **self.patient_info_entities,
            "organisation": str(self.organisation),
            "referral_reason": replace_delimiter(self.referral_reason),
            "diagnostic_conclude": replace_delimiter(self.diagnostic_conclude),
            "overall_inter": replace_delimiter(self.overall_inter),
            "pos_neg_result": self.pos_neg_result,
            "recd_recommen": replace_delimiter(self.recd_recommen),
            "hgvs_desc": replace_delimiter(self.hgvs_desc),
            "panelapp_url": join_list(self.panelapp_url),
            "pland_testmethod": replace_delimiter(self.pland_testmethod),
            "gene_list": join_list(self.gene_list),
        }
        if self.pos_neg_result:
            self.final_result_summary.update(
                get_filtered_variants(self.variants_identified)
            )



class TableExtractor:
    """
    This class is the base class that processes the textual content to detect tables. Table headers are detected based on a given set of headers and using word embedding vectors for similarity. This can detect simple tables present within genomics reports.
    """

    def __init__(self):
        self.tableconfig = TableExtractionConfig()

    def _get_tables(self) -> list:
        """
        Detects tables and extracts the tables based on header types

            Parameters: None

        """
        self.tables = [table for page in self.pages for table in page.extract_tables()]
        processed_tables = []
        for table in self.tables:
            temp_table = {}
            header_types = self._get_table_header_type(table)
            if len([v for h in header_types for k, v in h.items() if v != 0]) == 1:
                temp_table[
                    [k for h in header_types for k, v in h.items() if v != 0][0]
                ] = table
                processed_tables.append(temp_table)
        return processed_tables

    def _get_table_header_type(self, table: list) -> list:
        """
        Given a list of tables detected, header types are classified and list of dict(table_header, table) returned.

            Parameters:
                table(list): list of rows from a table

            Returns:
                list of header types and output

        """
        # header_type = patient_info, report_authorisation, variant_info, panel_and_coverage
        def find_patterns_key_value(value: list) -> bool:
            """
            find the patterns based on the regex pattern provided.
            """
            regex = re.compile(self.tableconfig.key_value_pair_regex, re.VERBOSE)
            text_content = " ".join([val for val in value if val != None])
            matches = regex.findall(text_content)
            return True if len(matches) else False

        def keyword_matches(keywords: list, value: list) -> int:
            """
            find whether keywords are present in a text.
            """
            value = [v for v in value if v]
            return len(
                [
                    True
                    for key in keywords
                    if key in " ".join(list(map(lambda x: x.lower(), value)))
                ]
            )

        # specific to the type of tables present, need to generalise it based on other reports
        def contains_patient_info(
            table: list,
            threshold: int = 1,
            keywords: dict = self.tableconfig.keywords_patient_info,
        ) -> dict:
            """
            check whether the table header represents patient information.
            """
            return {
                self.tableconfig.patient_info: len(
                    [
                        True
                        for value in table
                        if find_patterns_key_value(value)
                        if keyword_matches(keywords, value)
                        > self.tableconfig.patient_info_threshold
                    ]
                )
            }

        def contains_report_info(
            table: list,
            threshold: int = 0,
            keywords: dict = self.tableconfig.keywords_report_info,
        ) -> dict:
            """
            check whether the table header represents the report information.
            """
            return {
                self.tableconfig.report_info: len(
                    [
                        True
                        for value in table
                        if find_patterns_key_value(value)
                        if keyword_matches(keywords, value)
                        > self.tableconfig.report_info_threshold
                    ]
                )
            }

        def variant_info(
            table: list,
            threshold: int = 2,
            keywords: dict = self.tableconfig.keywords_variant_info,
        ) -> dict:
            """
            check whether the table header represents variant information.
            """
            return {
                self.tableconfig.variant_info: len(
                    [
                        True
                        for value in table
                        if len(value) < 7
                        if keyword_matches(keywords, value)
                        >= self.tableconfig.variant_info_threshold
                    ]
                )
            }

        def panel_and_coverage(
            table: list,
            threshold: int = 1,
            keywords: dict = self.tableconfig.keywords_panel_coverage,
        ) -> dict:
            """
            check whether the table header represents panel and coverage information
            """
            return {
                self.tableconfig.panel_and_coverage: len(
                    [
                        True
                        for value in table
                        if value[0] is not None
                        if keyword_matches(keywords, [value[0]])
                        == self.tableconfig.panel_and_coverage_threshold
                    ]
                )
            }

        return [
            contains_patient_info(table),
            contains_report_info(table),
            variant_info(table),
            panel_and_coverage(table),
        ]


class SectionExtractor:
    """
    This class is the base class that processes the textual content by detecting the sections.
    Section detection is based on basic lingistic properties and a rule-based approach.
    Properties are:
    1. check if the potential text is not greater than N number of words (N is a threshold; default value = 4)
    2. check if the potential text has only nouns as words or the number of nouns present is greater than the rest.
    3. Given a list of potential section titles, check using exact word matching.
    4. Given a list of potential section titles, check using word embedding vectors for similarity.
    """

    def __init__(self):
        self.sectionconfig = SectionExtractionConfig()
        self.orgconfig = OrganisationTemplateConfig()
        self.section_titles = {}
        self.section = {}
        self.section_title_map = self.sectionconfig.section_title_map
        self.raw_text_combined = None
        self.raw_text = {}
        self.intermediate_keyvalues = []

    def __check_first_letter_cap(self, line: str) -> bool:
        """
        check whether the first letter in a string or text is a capitalized letter.

            Parameters:
                line(str) - text string

            Returns:
                True or False

        """
        words = nltk.word_tokenize(line)
        if words:
            if words[0][0].isupper():
                return True
        else:
            return False

    def __check_num_words(self, line: str, threshold: int) -> bool:
        """
        Check whether the number of tokenized words in a given text is less than the threshold value.

            Parameters:
                line(str): input string
                threshold(int): threshold for the length of words

            Returns:
                True or False

        """
        words = nltk.word_tokenize(line)
        if len(words) <= threshold:
            return True
        else:
            return False

    def __is_nouns_majority(self, line: str) -> bool:
        """
        Check whether the majority of words in a given text are noun phrases based on a threshold value.

            Parameters:
                line(str): input text

            Returns:
                True or False

        """
        words = nltk.word_tokenize(line)
        if len(words) - len(
            [
                w
                for w, tag in nltk.pos_tag(words)
                if w.isalpha()
                if self.sectionconfig.nn_exp in tag
            ]
        ) <= int(len(words) / 2.0):
            return True
        else:
            return False

    def _get_section_titles_patterns(self, content: str):
        """
        Based on the linguistic patterns, extract the section titles without any prior title information.

            Parameters:
                content(str): content from the document
        """
        self.section_titles = {
            line_num: line.strip()
            for line_num, line in enumerate(content)
            if self.__check_num_words(line)
            if self.__is_nouns_majority(line)
            if self.__check_first_letter_cap(line)
        }
        self.section_titles = remove_empty_keys_dict(self.section_titles)

    def _get_section_titles_exact_match(
        self, content: str, diagnosis: str, organisation: str
    ):
        """
        Based on a given set of keywords for each section title (prior information provided), identify the section titles and line numbers.

            Parameters:
                content(str): content from the document
                diagnosis(str): diagnosis present in the document
                organisation(str): organisation name present in the document
        """

        def check_title_value(line: str, list_of_values: list) -> bool:
            """
            check whether the keywords for a title is provided in the line.
            """
            chk_val = False
            for word in list_of_values:
                if word in line.lower():
                    chk_val = True
            return chk_val

        def check_title_func_value(line: str, func: list) -> bool:
            """
            check the title and see if it matches prior information provided.
            """
            chk_val = check_title_value(line, func)
            return chk_val

        chks_order = {
            "ref_reason": False,
            "summary": False,
            "inter": False,
            "implication": False,
            "further test": False,
            "info": False,
            "gene_information": False,
        }
        
        section_config = get_dict_based_organisation(self.orgconfig, organisation)
       
        threshold = self.sectionconfig.line_threshold

        for line_num, line in enumerate(content):
            delimiter = ":"
            if delimiter in line:
                line = line.split(delimiter)[0]
            if self.__check_num_words(line, threshold):
                split_line = line
                if self.__check_first_letter_cap(line):
                    # result summary
                    if not chks_order["summary"]:
                        if check_title_func_value(line, section_config["res_sum"]):
                            self.section_titles[
                                line_num
                            ] = self.orgconfig.section_title_map["res_sum"]
                            chks_order["summary"] = True
                    # result/interpretation
                    if not chks_order["inter"]:
                        if "summary" not in line.lower():
                            if check_title_func_value(
                                " ".join(line.split()[:2]), section_config["result"]
                            ):
                                self.section_titles[
                                    line_num
                                ] = self.orgconfig.section_title_map["result"]
                                chks_order["inter"] = True

                    # further testing
                    if check_title_func_value(line, section_config["further_test"]):
                        self.section_titles[
                            line_num
                        ] = self.orgconfig.section_title_map["further_test"]
                        chks_order["further test"] = True
                    
                    # gene information in cns
                    if check_title_func_value(
                            line, section_config["gene_information"]
                        ):
                            if not chks_order["gene_information"]:
                                self.section_titles[
                                    line_num
                                ] = self.orgconfig.section_title_map["gene_information"]
                                chks_order["gene_information"] = True

            else:
                split_line = " ".join(line.split()[:3])
            # referral reason
            if check_title_func_value(split_line, section_config["ref_reason"]):
                
                if not chks_order["ref_reason"]:
                    self.section_titles[line_num] = self.orgconfig.section_title_map[
                        "ref_reason"
                    ]
                    chks_order["ref_reason"] = True

            if check_title_func_value(line, section_config["test_information"]):
                if (
                        len(
                            [
                                True
                                for x in ["technical", "testing"]
                                if x in line.lower()
                            ]
                        )
                        == 0
                    ):
                        self.section_titles[
                            line_num
                        ] = self.orgconfig.section_title_map["test_information"]
                        chks_order["info"] = True 
                else:
                    self.section_titles[line_num] = self.orgconfig.section_title_map[
                        "test_information"
                    ]
                    chks_order["info"] = True

        self.section_titles = remove_empty_keys_dict(self.section_titles)

    def _get_section_titles_semantic_search(self):
        # need to add code here
        # not required for the genomics reports as semantic interpretation of section titles is not relevant.
        pass

    def __postprocess_sections_exact_match(self):
        """
        Postprocess information in section contents.
        """
        # use this section for postprocessing and cleaning the results
        pass

    def _get_sections(self, organisation, delimiter="\n"):
        """
        Extract the section titles and the corresponding content for each section title.
        """

        def get_key_values(each_inp, get_all_keys):
            kv = {}
            for num, w in enumerate(each_inp.split()):
                if w in get_all_keys:
                    if num != 0:
                        kv[w] = num - 1
                    else:
                        kv[w] = num
            return kv

        def get_multiple_keyvaluepairs(each_inp, rex):
            """
            Extract multiple key value pairs present
            """
            rex_pat = re.compile(r"[ ]*(.+):[ ]*\n?((?:.+\n?[^\s])*)", re.VERBOSE)
            get_all_keys = rex.findall(each_inp)

            kv = get_key_values(each_inp, get_all_keys)

            test_list = list(kv.values())
            res_list = list(zip(test_list, test_list[1:] + test_list[:1]))

            if len(res_list) > 1:
                self.intermediate_keyvalues.extend(
                    [
                        flatten_lists(
                            rex_pat.findall(" ".join(each_inp.split()[r[0] : r[1] + 1]))
                        )
                        for r in res_list
                    ]
                )
                self.intermediate_keyvalues.extend(
                    rex_pat.findall(" ".join(each_inp.split()[res_list[-1][0] :]))
                )
            if len(res_list) == 1:
                self.intermediate_keyvalues.extend(
                    rex_pat.findall(" ".join(each_inp.split()[res_list[0][0] :]))
                )

        content = self.raw_text_combined.split(delimiter)
        content = remove_introduction_patterns(content)
        self._get_section_titles_exact_match(content, self.diagnosis, self.organisation)
        section_line_num = list(self.section_titles.keys())
       
        # get personal info in case no tabular content present
        if section_line_num:
            introduction_content = content[0 : section_line_num[0]]
        else:
            introduction_content = []
        rex = re.compile(r"\w+(?: \w+)*:", re.VERBOSE)
        res = list(filter(lambda item: item not in [None], introduction_content))

        def get_date_index(content):
            break_num = 0
            for num, letter in enumerate(content):
                if letter.isnumeric():
                    if break_num == 0:
                        break_num = num
            return content[0:break_num], content[break_num:]

        for each_item in res:
            if "\n" in each_item:
                split_sentences = each_item.split("\n")
                for each_inp in split_sentences:

                    get_multiple_keyvaluepairs(each_inp, rex)
            else:
                get_multiple_keyvaluepairs(each_item, rex)

        for i in range(0, len(section_line_num)):
            start_pos = section_line_num[i]
            temp = i + 1
            if temp < len(section_line_num):
                end_pos = section_line_num[temp]
            else:
                end_pos = len(content)

            self.section[self.section_titles[section_line_num[i]]] = delimiter.join(
                content[start_pos:end_pos]
            )
            self.section = remove_empty_keys_dict(self.section)

        self.__postprocess_sections_exact_match()


class KeyValueExtractor:
    """
    Identify key-value pairs from tabular structure based on the alignment of key-value pairs.
    Identify key-value pairs based on regex rule.
    """

    def __init__(self):
        self.keyvalueconfig = KeyValuePairsConfig()

    def _compare_values(self, value: str, list_of_keys: list) -> bool:
        """
        check whether a key is present in the text content

            Parameters:
                value(str): input value
                list_of_keys(list): list of values for comparison

            Returns:
                val_present: True or False
        """
        val_present = False
        for val in list_of_keys:
            if value:
                if val in value.lower():
                    val_present = True
        return val_present

    def _extract_patient_info_section(self, content: str, mode: str = "exact") -> dict:
        """
        extract patient information based on exact matches of keys

            Parameters:
                content(str): text content
                mode(str): requires an exact match

            Returns:
                patient_pairs: dict with key-values representing patient information

        """
        if mode == "exact":
            patient_pairs = {}
            for keyval in content:
                for key, values in self.keyvalueconfig.keywords_patient_info.items():
                    if len(keyval) == 2:
                        if self._compare_values(keyval[0], values):
                            patient_pairs[key] = keyval[1]
                            # patient_pairs[key] = ''
            return patient_pairs

    def _extract_patient_info_tables(self, table: list, mode: str = "exact") -> dict:
        """
        extract patient information from tables based on exact key matches.

            Parameters:
                table(list): list containing tabular rows
                mode(str): requires an exact match

            Returns:
                patient_pairs: dict with key-values representing patient information
        """

        if mode == "exact":
            patient_pairs = {}
            temp_values = []
            potential_key_values = flatten_lists(table)
            rex = re.compile(
                self.keyvalueconfig.regex_expression_entity_value, re.VERBOSE
            )
            res = list(filter(lambda item: item not in [None], potential_key_values))
            for each_item in res:
                if "\n" in each_item:
                    split_sentences = each_item.split("\n")
                    temp_values.extend(
                        [rex.findall(each_inp) for each_inp in split_sentences]
                    )
                else:
                    temp_values.extend(rex.findall(each_item))
            temp_values = flatten_lists(temp_values)
            for keyval in temp_values:
                if type(keyval) is tuple:
                    for (
                        key,
                        values,
                    ) in self.keyvalueconfig.keywords_patient_info.items():
                        if self._compare_values(keyval[0], values):
                            patient_pairs[key] = keyval[1]
                            # patient_pairs[key] = ''
            return patient_pairs

    def _extract_variant_tables(self, table: list, mode: str = "exact") -> list:
        """
        extract information from variant tables based on exact key matches

            Paramters:
                table(list): list of rows in a table
                mode(str): requires an exact match

            Returns:
                list of dictionary containing variant information from tables
        """

        def change_table_features(std_key, table_row, key):
            if key is None:
                return {key: table_row}
            if key.lower() == "analysis":
                return {"gene": table_row.split()[0]}
            elif key.lower() == "result":
                hgvs = ""
                inheritance = ""
                for word in table_row.split():
                    if "c." in word:
                        hgvs = hgvs + " " + word
                    if "p." in word:
                        hgvs = hgvs + " " + word
                    if word.lower() in self.keyvalueconfig.zygosity.keys():
                        inheritance = word
                return {"hgvs": hgvs, "interitance": inheritance}
            else:

                return {std_key: table_row}

        if mode == "exact":
            variant_pairs = []
            header = [
                h.replace("\n", " ") for h in table[0] if h is not None
            ]  # replace \n with a space

            for table_num, table_content in enumerate(table[1:]):
                temp_variant = {}
                for key, values in self.keyvalueconfig.keywords_variant_info.items():
                    for num, h in enumerate(header):
                        if self._compare_values(h, values):
                            temp_variant.update(
                                change_table_features(
                                    key, table[table_num + 1][num], table[0][num]
                                )
                            )

                            # temp_variant[key] = table[table_num + 1][num]
                variant_pairs.append(temp_variant)
            return variant_pairs

    def _extract_panel_app_tables(self, table: list, mode: str = "exact") -> dict:
        """
        extract panel app information from tables.
        """
        if mode == "exact":

            def check_if_splitter_present(row: str, delimiter: str = ":") -> bool:
                """
                check is delimiter is present or not.
                """
                if delimiter in row[0]:
                    return True
                else:
                    return False

            panel_apps = {}
            if len(table[0]) == 1:
                if "panel" in table[0][0].lower():
                    panel_apps["gene list"] = table[1][0]

            elif len(table[0]) == 1 and len(table) >= 2:
                chk_first_row_not_gene = True
                for num, row in enumerate(table):
                    if check_if_splitter_present(row):
                        for (
                            key,
                            values,
                        ) in self.keyvalueconfig.keywords_panel_app_info.items():
                            if self._compare_values(row[0].split(":")[0], values):
                                panel_apps[key] = [row[0].split(":")[1]]
                                if num == 0 and key == "gene list":
                                    chk_first_row_not_gene = False
                if (
                    "clinical indications" not in panel_apps.keys()
                    and chk_first_row_not_gene
                ):
                    # check if first row is not gene list or other options
                    # check if it is not a panel list
                    panel_apps["clinical indications"] = table[0][0]

            if len(table[0]) > 1:
                panel_apps = {}
                for row in table:
                    for (
                        key,
                        values,
                    ) in self.keyvalueconfig.keywords_panel_app_info.items():
                        if self._compare_values(row[0], values):
                            panel_apps[key] = [r for r in row[1:] if r is not None]
            return panel_apps


class DocClass:
    """
    This class is used for classification of documents as MR vs scanned, genomic reports vs others.
    """

    def __init__(self, doc_path: str):
        assert type(doc_path) == str, "Path is not string format"
        self.doc_path = doc_path
        self.is_mr: bool = None
        self.docclassconfig = DocClassConfig()

    def _check_is_mr(self) -> bool:
        """
        Check whether the given document is machine readable or not.

            Parameters: None

            Returns:
             True or False
        """
        return bool(
            [True if len(page.extract_text()) else False for page in self.pages][0]
        )

    def _get_pages(self):
        """
        Reads the PDF using PDFPlumber package and returns the pages.

            Parameters: None

        """
        # get the pages using pdfplumber package
        pdf_file = pb.open(self.doc_path)
        self.pages = pdf_file.pages

    def _raw_text_content(self):
        """
        Extracts the raw text using PDFPlumber package per page, and combines them into a single text.

            Parameters: None
        """
        # extract the raw content per page and combine them per document as a string
        self.raw_text = {
            num: page.extract_text() for num, page in enumerate(self.pages)
        }
        self.raw_text_combined = "\n".join(self.raw_text.values())

    def _check_diagnosis_present(self) -> str:
        """
        Checks if a diagnosis is present and returns the value; else returns an empty string.

            Parameters: None

        """

        def chk_val_from_list(list_of_vals, text_content):
            output_list = [
                each_val
                for each_val in list_of_vals
                if each_val.lower()
                in text_content.lower().replace(",", " ").replace(".", " ")
            ]
            if output_list:
                return True
            else:
                return False

        def extraction_diagnosis_list(diagnosis_config):
            return [
                diagnosis
                for diagnosis, list_of_vals in diagnosis_config.items()
                if chk_val_from_list(list_of_vals, self.raw_text_combined) == True
            ]

        # check whether diagnosis is present in the document and if so, return the corresponding diagnosis
        diagnosis_present = []
        # we follow the following steps as there are overlaps between keywords representing different diagnosis
        # first we check for tumour
        if len(diagnosis_present) == 0:
            diagnosis_present = extraction_diagnosis_list(
                self.docclassconfig.diagnosis_tumour
            )
        # next we check for muscular dystrophy
        if len(diagnosis_present) == 0:
            diagnosis_present = extraction_diagnosis_list(
                self.docclassconfig.diagnosis_md
            )
        # next we check for other diagnosis
        if len(diagnosis_present) == 0:
            diagnosis_present = extraction_diagnosis_list(self.docclassconfig.diagnosis)

        if len(diagnosis_present) == 0:
            diagnosis_present = [
                diagnosis
                for diagnosis, list_of_vals in self.docclassconfig.diagnosis_neuroblastomas.items()
                for each_val in list_of_vals
                if len(
                    [
                        each_val.lower()
                        for w in self.raw_text[0]
                        .lower()
                        .replace(",", " ")
                        .replace(".", " ")
                        .split()
                        if each_val.lower() in w
                    ]
                )
                > 1
            ]

        if diagnosis_present:
            return diagnosis_present[0]
        else:
            return ""

    def _check_report_status(self):
        """
        checks whether a diagnosis is present and if present, whether the report is a genomic report or not.

            Parameters: None
        """
        # check the status of the document, whether it is a genomic report or not, return the output along with the diagnosis information
        is_report = False
        diagnosis_present = ""
        for tokens in self.docclassconfig.fuzzymatchdoc:
            if (
                tokens
                in str(self.raw_text_combined)
                .replace(",", " ")
                .replace(".", " ")
                .lower()
            ):
                diagnosis_present = self._check_diagnosis_present()
                if diagnosis_present != "":
                    is_report = True

        return is_report, diagnosis_present

    def _organisation_name(self) -> str:
        """
        checks if organisation name is present and returns the corresponding number.

            Parameters: None
        """
        # extract the name of the organisation if present in the document
        def compare_values(list_of_values, content):
            is_present = False
            for val in list_of_values:
                if val.lower() in " ".join(content.lower().split("\n")[:20]):
                    is_present = True
            return is_present

        organisation_num = [
            num
            for num, list_of_org_names in self.docclassconfig.organisation.items()
            if compare_values(list_of_org_names, self.raw_text_combined)
        ]

        if len(organisation_num) > 1:
            organisation_num = [num for num in organisation_num if num != "1"]

        if organisation_num:
            return organisation_num[0]
        else:
            return "0"

    def document_classification_per_file(self) -> dict:
        """
        Classify documents as MR and scanned. Further MR docs are classified as genomic reports vs other.
        Extract the diagnosis and organisation information of the document.

            Parameters: None
        """
        file_path = os.path.split(self.doc_path)[-1]
        self._get_pages()
        is_mr = self._check_is_mr()

        if is_mr:
            self._raw_text_content()
            is_report, diagnosis_present = self._check_report_status()
            num = self._organisation_name()
        else:
            is_report = "NA"
            diagnosis_present = "NA"
            num = "NA"

        return {
            "is_machine_readable": is_mr,
            "filename": file_path,
            "is_genomic_report": is_report,
            "diagnosis": diagnosis_present,
            "organisation": num
        }


class DocSpec(
    DocClass, TableExtractor, SectionExtractor, KeyValueExtractor, ReportStructure
):
    """
    This class inherits the base classes TableExtractor and SectionExtractor and represents the document information such as whether this document is machine-readable or
    scanned, whether tabular content or textual information is present, number of pages, number of tables,
    extracted text and tabular content.
    """

    def __init__(self, doc_path: str, diagnosis: str, organisation: str):
        assert type(doc_path) == str, "Path is not in string format"

        self.doc_path: str = doc_path
        self.diagnosis: str = diagnosis
        self.organisation: str = organisation
        self.number_of_pages: int = None
        self.number_of_tables: int = None
        self.text_present: int = None
        self.tables_present: bool = None

        TableExtractor.__init__(self)
        SectionExtractor.__init__(self)
        KeyValueExtractor.__init__(self)
        DocClass.__init__(self, doc_path)

    def __check_tables(self) -> int:
        """
        find whether tables are present or not

            Parameters: None

        """
        return sum([len(page.find_tables()) for page in self.pages])

    def _get_meta_info(self):
        """
        Extract all the meta information of the document.

            Parameters: None

        """
        # get meta data on the document
        self._get_pages()
        self.number_of_pages = len(self.pages)
        self.number_of_tables = self.__check_tables()
        self.tables_present = True if self.number_of_tables > 0 else False
        self.is_mr = self._check_is_mr()
        self._raw_text_content()
        self.text_present = True if len(self.raw_text_combined.split("\n")) else False

    def _get_section_info(self):
        """
        Get section information for gene variants.

            Parameters: None

        """
        self._get_sections(self.organisation)
    

    def _extract_entities_tables(self, tables: list):
        """
        Extract key-value pairs from the different tables.

            Parameters: None

        """
        variant_pairs = []
        panel_apps = {}
        patient_pairs = {}
        for each_table in tables:
            for key, table_content in each_table.items():
                if key == self.tableconfig.variant_info:
                    variant_pairs.extend(self._extract_variant_tables(table_content))
                if key == self.tableconfig.panel_and_coverage:
                    panel_apps = self._extract_panel_app_tables(table_content)
                if key == self.tableconfig.patient_info:
                    patient_pairs = self._extract_patient_info_tables(table_content)

        if len(panel_apps) == 0:
            temp_list = []
            temp_gene_list = []
            for each_table in tables:
                for key, table_content in each_table.items():
                    if key == self.tableconfig.panel_and_coverage:
                        if (
                            "panel"
                            in flatten_lists(
                                [
                                    t.lower().split()
                                    for t in table_content[0]
                                    if t is not None
                                ]
                            )
                            and len(table_content) == 2
                        ):  
                            temp_list.append(table_content[0])
                            temp_gene_list.append(table_content[1])
            panel_apps["panel apps and versions"] = temp_list
            panel_apps["gene list"] = temp_gene_list

        return variant_pairs, panel_apps, patient_pairs

    def extract_report_information(self):
        """
        Information from the report extracted

            Parameters: None

        """
        self._get_meta_info()
        self._get_section_info()
        tables = self._get_tables()
        var_pairs, panel_apps, patient_pairs = self._extract_entities_tables(tables)
        if len(patient_pairs) == 0:
            patient_pairs = self._extract_patient_info_section(
                self.intermediate_keyvalues
            )

        return ReportStructure(
            self.doc_path,
            self.diagnosis,
            self.organisation,
            self.number_of_pages,
            self.number_of_tables,
            self.text_present,
            self.tables_present,
            self.is_mr,
            self.raw_text,
            self.section,
            tables,
            [],
            var_pairs,
            panel_apps,
            patient_pairs,
            []
        )
