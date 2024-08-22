# General utility functions

import re
import itertools
from nltk import sent_tokenize
import numpy as np
import pandas as pd

from docprocessing.config import OntologyMatcher, KeyValuePairsConfig
from pretrained_word_embeddings.embeddings import SentenceEmbeddings
from pretrained_word_embeddings import config as SentenceEmbedConfig


def find_patterns_key_value(text_content, pattern=None):
    """
    return matches from a text content based on the regex pattern provided.

        Parameters:
            text_content: input text for searching the pattern
            pattern: regex pattern for search

        Returns:
            list of matching text phrases
    """
    if pattern is None:
        pattern = r"([a-zA-Z0-9 ]*:[a-zA-Z0-9\-/ :]*)"
    text_content = str(text_content)
    regex = re.compile(pattern, re.VERBOSE)
    matches = regex.findall(text_content)
    return matches


def find_dates(
    text_content,
    pattern=None,
):
    """
    Find the dates using regex pattern

        Parameters:
            text_content: input text for searching the regex pattern
            pattern: regex pattern for search
        Returns:
            Return the date if present

    """
    if pattern is None:
        pattern = r"(\d{1,2}[-/]\d{1,2}[/-]\d{2,4})|(\d{1,2}/\d{4})|((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ -.]*\d{2}[thsdn, .-]*\d{4})|((?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,. ]*\d{4})|(\d{4})"
    matches = find_patterns_key_value(text_content, pattern)
    if matches:
        return matches[0]
    else:
        return ""


def find_gender(text_content):
    """
    Extract the gender information from text

        Parameters:
            text_content: text string that contains information whether male or female or it is missing
        
        Returns:
            returns whether it is M, F or Missing
    """
    if "female" in text_content.lower():
        return "F"
    elif "male" in text_content.lower():
        return "M"
    else:
        return "Missing"



def flatten_lists(input_lists):
    """
    Given a list containing list of lists, flatten it into a single list.
    """
    return list(itertools.chain.from_iterable(input_lists))


def join_list(list_of_items):
    """
    given a list of items, return the list joined using ',' as a single string.
    """
    if type(list_of_items) == list:
        try:
            return replace_delimiter(",".join(list_of_items))
        except:
            return list_of_items
    elif type(list_of_items) == str:
        return list_of_items
    else:
        return ""


def remove_white_spaces(word):
    """
    remove whitespaces present in text.
    """
    return word.replace(" ", "")


def replace_delimiter(content, delimiter="\n", replacewith=" "):
    """
    replace a delimiter in a text.
    """
    return content.replace(delimiter, replacewith)


def remove_empty_keys_dict(input_dict):
    """
    given a dict, remove key value pairs that have an empty value.
    """
    empty_keys = [k for k, v in input_dict.items() if v in [" ", "", "\n"]]
    for k in empty_keys:
        del input_dict[k]
    return input_dict

def remove_introduction_patterns(content):
    """
    Remove blocks of duplicated patient information from pages.
    """
    # assuming the patient information is present in the first 10 lines in the first page
    block_content = content[:10]
    removed_duplicates = [
        content_line
        for content_line in content[11:]
        if content_line not in block_content
    ]
    block_content.extend(removed_duplicates)
    return block_content


def get_gene_information(gene, section, tagger, is_screening_info_present="screening"):
    """
    Given a gene name, identify the gene information present in a section content.

    This captures the pattern where a gene name is followed by some information about it within a given section (assuming this section has been identified)
    """
    sent_found = section
    gene_found = False
    if section:
        for ind, sent in enumerate(sent_tokenize(section.replace("\n", " "))):
            if is_screening_info_present not in sent:
                for num, word in enumerate(sent.split()):
                    if gene_found == False:
                        if gene in word:
                            sent_split = " ".join(sent.split()[num + 1 :])
                            keys = [
                                key
                                for key, value in tagger.print_entities_per_sent(
                                    sent_split
                                ).items()
                                if value == "Gene"
                                if key != gene
                            ]

                            if not keys:
                                gene_found = True
                                sent_found = sent_split

    return sent_found


def get_inheritance_information(gene, section):
    """
    Given a gene and a corresponding section, identify inheritance information
    """
    ontology = OntologyMatcher()
    inheritance = ""
    if section:
        for sent in sent_tokenize(section.replace("\n", " ")):
            inheritance_found = False
            if inheritance_found == False:
                for key in ontology.inheritance_loinc.keys():
                    if [True for word in sent.split() if key.lower() == word.lower()]:
                        if gene in sent:
                            inheritance = ontology.inheritance_loinc[key]
                            inheritance_found = True
    return inheritance

def check_pos_neg_variants_from_summary(summary):
    """
    Based on the result summary, check whether we need to assign 1 or 0 for the pos_neg result. 

    Pos_neg represents whether the overall result is considered as a positive result or a negative result. The positive result can either be variants of clinical significance or not.
    """
    keyconfig = KeyValuePairsConfig()
    sent_emb = SentenceEmbeddings(
        SentenceEmbedConfig.MODEL_CLASSIFICATION,
        SentenceEmbedConfig.MODEL_THRESHOLD_CLASSIFICATION,
    )

    is_key_negative = False

    try:
        for key, value in keyconfig.key_sentences_no_variants_reported.items():
            if key == "negative":
                get_pred = []
                for phrase in value:
                    get_pred.extend(sent_emb.get_sentences(phrase, summary))
                    if get_pred:
                        is_key_negative = True
        
        if is_key_negative:
            return 0
        else:
            return 1 
        
    
    except:
        return 1


def get_variant_conclusion(conclusion):
    """
    Based on the conclusion text, extract the variant conclusion information.
    """
    keyconfig = KeyValuePairsConfig()
    ontology = OntologyMatcher()
    sent_emb = SentenceEmbeddings(
        SentenceEmbedConfig.MODEL_CLASSIFICATION,
        SentenceEmbedConfig.MODEL_THRESHOLD_CLASSIFICATION,
    )
    is_key_uncertain = False
    is_key_pathogenic = False

    try: 
        for key, value in keyconfig.key_sentences_classification.items():
            if key == "Uncertain significance":
                get_pred = []
                for phrase in value:
                    get_pred.extend(sent_emb.get_sentences(phrase, conclusion))
                if get_pred:
                    is_key_uncertain = True
            if key == "Pathogenic" and is_key_uncertain == False:
                get_pred = []
                for phrase in value:
                    get_pred.extend(sent_emb.get_sentences(phrase, conclusion))

                if get_pred:
                    is_key_pathogenic = True
        if is_key_pathogenic:
            return ontology.variant_classification["Pathogenic"]
        if is_key_uncertain:
            return ontology.variant_classification["Uncertain significance"]
    except:
            return "Cannot identify from text"


def get_variants(
    raw_text, diagnosis, organisation, gene_list, section, overall_inter, tagger
):
    """
    Given a gene list, identify all the variants and their information provided.
    """
    keypairsconfig = KeyValuePairsConfig()
    ontology = OntologyMatcher()
    gene_variants = []
    
    for sent in sent_tokenize(raw_text.replace("\n", " ")):
            if diagnosis == "cystic_fibrosis":
                sent = sent.lower().replace("cystic fibrosis", "CFTR")
                sent = sent.lower().replace("cf4v2", "CFTR")
                sent = sent.lower().replace("cfeu2", "CFTR")
                
            for gene in gene_list:
                zygosity = ""
                dna_change = ""
                amino_change = ""
                diagnosis_condition = True
                if gene in sent:
                    splitted = sent.split(gene)[1].split()
                    for key in keypairsconfig.zygosity.keys():
                        if key in sent.lower():
                            zygosity = ontology.zygosity[keypairsconfig.zygosity[key]]
                    if diagnosis == "cystic_fibrosis":
                        dna_change_list = [s for s in sent.split() if "c." in s]
                        amino_change_list = [s for s in sent.split() if "p." in s]
                    
                    else:
                        dna_change_list = [s for s in splitted if "c." in s]
                        amino_change_list = [s for s in splitted if "p." in s]

                    if dna_change_list:
                        dna_change = dna_change_list[0]
                    if amino_change_list:
                        amino_change = amino_change_list[0]

                    if diagnosis == "sma" and overall_inter:
                        # for sma
                        if dna_change == "":
                            if gene in overall_inter:
                                split_sec = overall_inter.split(gene)[1]
                                if "SMN" in split_sec:
                                    dna_change = split_sec.split("SMN")[0]
                                else:
                                    dna_change = split_sec
                            if gene in ["DMD"]:
                                for each_sent in sent_tokenize(overall_inter):
                                    if "exon" in each_sent:
                                        dna_change = each_sent

                    classification = get_variant_conclusion(overall_inter)
                    if diagnosis == "epilepsy":
                        if len(dna_change) == 0 and len(amino_change) == 0:
                            diagnosis_condition = False

                    if diagnosis_condition:
                            gene_variants.append(
                                {
                                    "gene_name": gene,
                                    "transcript_ref_id": "",
                                    "dna_change_id": dna_change,
                                    "amino_change_id": amino_change,
                                    "zygosity": zygosity,
                                    "gene_information": get_gene_information(
                                        gene, section, tagger
                                    ),
                                    "inheritance": get_inheritance_information(
                                        gene, section
                                    ),
                                    "classification": classification,
                                    "variant_evidence": "",
                                }
                            )

    return gene_variants


def get_transcripts_id(variants_dict, section):
    """
    For each variant identified, identify the transcript IDs present.
    """
    for each_dict in variants_dict:
        transcript = ""
        gene_name = each_dict["gene_name"]
        prev_NM = 0
        if section:
            for num, w in enumerate(section.split()):
                if "NM_" in w:
                    if prev_NM != 0:
                        if [
                            True
                            for word in section.split()[prev_NM:num]
                            if gene_name in word
                        ]:
                            transcript = w
                            prev_NM = num
                    else:
                        if [
                            True for word in section.split()[:num] if gene_name in word
                        ]:
                            transcript = w
                            prev_NM = num

                if transcript:
                    each_dict["transcript_ref_id"] = transcript
    return variants_dict


def get_filtered_variants(variants_identified):
    """
    Remove duplicates and convert variant information to our data model format.
    """
    # need to refactor this code in a better way!
    filtered_variants = {}
    data = pd.DataFrame(variants_identified)
    if "gene_name" in data.columns.values.tolist():
        #df_result = (
        #    data.groupby(["gene_name"])
        #    .agg(lambda val: [x for x in val.tolist() if x if str(x) != "nan"])
        #    .reset_index()
        #)
        df_result = data
        gene_names = df_result.gene_name.to_list()
        transcript_ref_id = df_result.transcript_ref_id.to_list()
        dna_change_id = df_result.dna_change_id.to_list()
        amino_change_id = df_result.amino_change_id.to_list()
        zygosity = df_result.zygosity.to_list()
        gene_information = df_result.gene_information.to_list()
        inheritance = df_result.inheritance.to_list()
        classification = df_result.classification.to_list()
        variant_evidence = df_result.variant_evidence.to_list()

        for num, g_name in enumerate(gene_names):
            filtered_variants["gene_name_{}".format(num + 1)] = g_name
            if transcript_ref_id[num]:
                filtered_variants[
                    "transcript_ref_id_{}".format(num + 1)
                ] = transcript_ref_id[num]
            else:
                filtered_variants["transcript_ref_id_{}".format(num + 1)] = ""
            if dna_change_id[num]:
                filtered_variants["dna_change_id_{}".format(num + 1)] = dna_change_id[
                    num
                ]
            else:
                filtered_variants["dna_change_id_{}".format(num + 1)] = ""
            if amino_change_id[num]:
                filtered_variants[
                    "amino_change_id_{}".format(num + 1)
                ] = amino_change_id[num]
            else:
                filtered_variants["amino_change_id_{}".format(num + 1)] = ""
            if zygosity[num]:
                filtered_variants["zygosity_{}".format(num + 1)] = zygosity[num]
            else:
                filtered_variants["zygosity_{}".format(num + 1)] = ""
            if gene_information[num]:
                filtered_variants[
                    "gene_information_{}".format(num + 1)
                ] = gene_information[num]
            else:
                filtered_variants["gene_information_{}".format(num + 1)] = ""
            if inheritance[num]:
                filtered_variants["inheritance_{}".format(num + 1)] = inheritance[num]
            else:
                filtered_variants["inheritance_{}".format(num + 1)] = ""
            if classification[num]:
                filtered_variants["classification_{}".format(num + 1)] = classification[
                    num
                ]
            else:
                filtered_variants["classification_{}".format(num + 1)] = ""
            if variant_evidence[num]:
                filtered_variants[
                    "variant_evidence_{}".format(num + 1)
                ] = variant_evidence[num]
            else:
                filtered_variants["variant_evidence_{}".format(num + 1)] = ""
    return filtered_variants


def get_panel_list_regex_pattern(section, pattern=r"^\d*[.,]?\d*$"):
    """
    Extract panel information based on regex pattern.
    """
    decimal = re.compile(pattern)

    panel_apps = []

    for sent in sent_tokenize(section.replace("\n", " ")):
        sent = sent.split(":")
        get_potential_app = [
            each_s
            for s in sent
            for each_s in s.split(",")
            for num, each_w in enumerate(each_s.split())
            if len(each_s) < 5
            if "v" in each_w
            if decimal.match(each_w.split("v")[1])
        ]
        
        for each_phrase in get_potential_app:
            for num, each_word in enumerate(each_phrase.split()):
                if "v" in each_word:
                    get_num = num
            if get_num is not None:
                panel_apps.append(" ".join(each_phrase.split()[: get_num + 1]))
    return list(set(panel_apps))


def get_classification_from_conc(conclusion):
    """
    Get classification information based on conclusion text.
    """
    keyconfig = KeyValuePairsConfig()
    ontology = OntologyMatcher()
    sent_emb = SentenceEmbeddings(
        SentenceEmbedConfig.MODEL_CONCLUSION,
        SentenceEmbedConfig.MODEL_THRESHOLD_CONCLUSION,
    )
    is_key_uncertain = False
    is_key_pathogenic = False
    for key, value in keyconfig.key_sentences_conclusion.items():

        if key == "Uncertain significance":
            get_pred = []
            for phrase in value:
                get_pred.extend(sent_emb.get_sentences(phrase, conclusion))
            if get_pred:
                is_key_uncertain = True
        if key == "Pathogenic" and is_key_uncertain == False:
            get_pred = []
            for phrase in value:
                get_pred.extend(sent_emb.get_sentences(phrase, conclusion))

            if get_pred:
                is_key_pathogenic = True

    if is_key_pathogenic:
        return ontology.variant_classification["Pathogenic"]
    if is_key_uncertain:
        return ontology.variant_classification["Uncertain significance"]


def get_dict_based_organisation(classvar, organisation_name: str):
    """
    Return the corresponding dict for an organisation containing the section titles and keywords.
    #here we are only using the GOSH template but this can be expanded
    """
    if str(organisation_name) == "1":
        return classvar.ORG_1
