"""
All configurations are provided in this file. 
"""
from dataclasses import dataclass, field

#Keywords used for a rule-based classification of tables - patient information, report information, variant information and gene panel
# this is limited and can be expanded to add more keywords
@dataclass(frozen=True)
class TableExtractionConfig:
    key_value_pair_regex: str = r"([a-zA-Z0-9 ]*:[a-zA-Z0-9\-/ :]*)"
    keywords_patient_info: list = field(
        default_factory=lambda: [
            "name",
            "dob",
            "gender",
            "forename",
            "surname",
            "date of birth",
        ]
    )
    keywords_report_info: list = field(
        default_factory=lambda: ["reported", "authorised", "checked"]
    )
    keywords_variant_info: list = field(
        default_factory=lambda: [
            "gene",
            "zygosity",
            "inheritance",
            "hgv",
            "classification",
            "episode",
            "sequence analysis",
            "conclusion",
            "result",
            "analysis",
        ]
    )
    # panel and coverage
    # gene panel name, gene size, gene panel version
    keywords_panel_coverage: list = field(
        default_factory=lambda: [
            "clinical indication",
            "panel",
            "average",
            "gene list",
            "coverage",
        ]
    )

    patient_info: str = "patient information"
    report_info: str = "report and authorisation"
    variant_info: str = "variant information"
    panel_and_coverage: str = "panel and coverage information"

    # threshold for comparing with keywords provided above, the limit on searching for keywords within a table header
    patient_info_threshold: int = 1
    report_info_threshold: int = 0
    variant_info_threshold: int = 2
    panel_and_coverage_threshold: int = 1

#keyphrases extracted for identifying key sections based on examples of templates. This can be expanded based on newer reports.
@dataclass(frozen=True)
class OrganisationTemplateConfig:
    section_title_map: dict = field(
        default_factory=lambda: {
            "ref_reason": "referral reason",
            "res_sum": "diagnostic_conclude",
            "result": "overall_inter",
            "implication": "implication",
            "further_test": "recd_recommen",
            "test_information": "plandtestmethod",
            "gene_information": "gene_information",
        }
    )
    
    #dummy test report based on GOSH template is provided as an example
    ORG_1: dict = field(
        default_factory=lambda: {
            "ref_reason": ["referral reason", "referral", "indication"],
            "res_sum": ["summary"],
            "result": ["result", "report", "comments", "direct sequencing", "interpretation", "variant interpretation and clinical correlation"],
            "test_information": ["test methodology", "test information", "notes", "real-time pcr array", "technical information"],
            "implication_result": ["implications"],
            "further_test": ["further testing", "recommendation", "further work"],
            "gene_information": ["further information"]
        }
    )

#keyphrases to map organisational information to different sections
@dataclass(frozen=True)
class SectionExtractionConfig:
    line_threshold: int = 3
    nn_exp: str = "NN"
    nnp_exp: str = "NNP"

    # section_titles
    ref_reason: list = field(
        default_factory=lambda: ["referral reason", "referral", "reason for testing"]
    )
    # val_test: list = field(default_factory=lambda : ['validation', 'test'])
    res_sum: list = field(default_factory=lambda: ["summary"])
    result: list = field(default_factory=lambda: ["result", "report", "comments"])
    implication_result: list = field(default_factory=lambda: ["implications"])
    further_test: list = field(default_factory=lambda: ["further testing"])

    test_information: list = field(
        default_factory=lambda: [
            "test information",
            "information",
            "notes",
            "test methodology",
        ]
    )
    #remove information for GOSH templates
    #this is not required for general cases 
    section_delimiter_further_test: str = "Reported by"
    section_delimiter_result: str = "technical information"
    section_delimiter_information: str = ".\n.\n."
    # predefined section titles and criteria
    section_title_map: dict = field(
        default_factory=lambda: {
            "ref_reason": "referral reason",
            "res_sum": "diagnostic_conclude",
            "result": "overall_inter",
            "implication": "implication",
            "further test": "recd_recommen",
            "test_information": "plandtestmethod",
        }
    )

#keywords for document classification and diagnosis
#this can be extended with more keyphrases
@dataclass(frozen=True)
class DocClassConfig:
    organisation: dict = field(
        default_factory=lambda: {
            "1": [
                "Great Ormond Street Hospital for Children NHS Foundation Trust",
                "Great Ormond Street Hospital"
            ]
        }
    )
    fuzzymatchdoc: list = field(
        default_factory=lambda: [
            "genomic laboratory",
            "laboratory report",
            "analysis report",
            "genetic analysis report",
            "genetics report",
            "referral reason",
            "test methodology",
            "evidence for classification",
            "further testing",
            "gene panel",
            "next generation sequence",
            "molecular genetics report",
            "molecular genetic summary",
            "panel report"
        ]
    )

    diagnosis: dict = field(
        default_factory=lambda: {
            "cystic_fibrosis": ["cftr", "cystic fibrosis"],
            "epilepsy": [
                "epilepsy",
                "infantile spasms",
                "developmental delay",
                "autosomal disorder",
                "epilept",
            ],
            "sma": ["smn1", "smn2", "spinal muscular atrophy"],
            "medullary_thyroid": [
                "congenital adrenal hyperplasia",
                "medullary thyroid",
            ],
        }
    )

    diagnosis_md: dict = field(
        default_factory=lambda: {"muscular dystrophy": ["muscular dystrophy"]}
    )

    diagnosis_neuroblastomas: dict = field(
        default_factory=lambda: {
            "neuroblastomas": ["NF1", "NF2", "nephropathy", "SPRED1"]
        }
    )
    diagnosis_tumour: dict = field(
        default_factory=lambda: {
            "solid childhood tumours": ["methylation", "solid tumour", "tumour content"]
        }
    )


@dataclass(frozen=True)
class KeyValuePairsConfig:
    regex_expression_entity_value: str = r"[ ]*(.+):[ ]*\n?((?:.+\n?[^\s])*)"
    fuzzymatchdoc = [
        "genomic laboratory",
        "laboratory report",
        "analysis report",
        "genetic analysis report",
        "genetics report",
        "referral reason",
        "test methodology",
        "evidence for classification", 
        "further testing",
        "gene panel",
        "next generation sequence",
    ]

    # potential keys for patient information
    keywords_patient_info: dict = field(
        default_factory=lambda: {
            "Patient Name": ["name"],
            "Forename": ["forename"],
            "Surname": ["surname"],
            "Date of Birth": ["dob", "date of birth", "birth"],
            "Gender": ["gender"],
            "NHS number": ["nhs number", "nhs no"],
            "MRN number": ["mrn number", "mrn", "your reference", "your ref"],
            "Family number": ["family number"],
        }
    )

    # potential keys for variant information
    keywords_variant_info: dict = field(
        default_factory=lambda: {
            "gene": ["gene"],
            "zygosity": ["zygosity"],
            "inheritance": ["inheritance"],
            "hgvs": [
                "hgvs",
                "human genome variation",
                "sequence analysis",
                "panel result",
                "analysis",
                "result",
            ],
            "location": ["location"],
            "conclusion": ["conclusion"],
            "classification": ["classification"],
            "transcript": ["transcript"],
            "cdna": ["cdna"],
            "protein": ["protein"],
        }
    )
    # potential keys for panel apps information
    keywords_panel_app_info: dict = field(
        default_factory=lambda: {
            "clinical indications": ["indication"],
            "panel apps and versions": ["panel apps", "version"],
            "gene list": ["gene list", "genes"],
        }
    )

    zygosity: dict = field(
        default_factory=lambda: {
            "heterozygo": "Heterozygous",
            "hemizygo": "Hemizygous",
            "homozygo": "Homozygous",
            "homoplasm": "Homoplasmic",
            "heteroplasm": "Heteroplasmic",
        }
    )

    # sentences for semantic search of classification

    key_sentences_classification: dict = field(
        default_factory=lambda: {
            "Pathogenic": [
                "the variant is considered as pathogenic",
                "the variant is considered as diagnosis confirmed",
            ],
            "Likely pathogenic": ["likely pathogenic is detected"],
            "Uncertain significance": [
                "the variant is considered as uncertain clinical significance",
                "the following heterozygous variants of uncertain clinical significance were detected",
                "the following homozygous variants of uncertain clinical significance were detected",
                "the following homozygous variant of uncertain clinical significance was detected",
            ],
            "Benign": ["the variant is considered as benign"],
        }
    )

    key_sentences_conclusion: dict = field(
        default_factory=lambda: {
            "Uncertain significance": [
                "diagnosis is not confirmed",
                "variant of uncertain clinical significance is confirmed",
            ],
            "Pathogenic": ["consistent with diagnosis", "diagnosis is confirmed"],
        }
    )

    key_sentences_no_variants_reported: dict = field(
        default_factory=lambda:
        {
            "negative":
            ["highly unlikely",
             "unlikely to be due to variants"]
        }
    )


@dataclass(frozen=True)
class OntologyMatcher:
    inheritance_loinc: dict = field(
        default_factory=lambda: {
            "Autosomal dominant": "LA24640-7|Autosomal dominant|http://loinc.org",
            "Autosomal recessive": "LA24641-5|Autosomal recessive|http://loinc.org",
            "Mitochondrial": "LA24789-2|Mitochondrial|http://loinc.org",
            "Isolated": "LA24949-2|Isolated|http://loinc.org",
            "X-linked": "LA24947-6|X-linked|http://loinc.org",
            "AD": "LA24640-7|Autosomal dominant|http://loinc.org",
            "AR": "LA24641-5|Autosomal recessive|http://loinc.org",
            "XL": "LA24947-6|X-linked|http://loinc.org",
            "XLD": "LA24947-6|X-linked|http://loinc.org",
            "XR": "LA24947-6|X-linked|http://loinc.org",
            "XD": "LA24947-6|X-linked|http://loinc.org",
            "Not known": "",
        }
    )

    variant_classification: dict = field(
        default_factory=lambda: {
            "Pathogenic": "LA6668-3|Pathogenic|http://loinc.org",
            "Likely pathogenic": "LA26332-9|Likely pathogenic|http://loinc.org",
            "Uncertain significance": "LA26333-7|Uncertain significance|http://loinc.org",
            "Benign": "LA6675-8|Benign|http://loinc.org",
            "Likely benign": "LA26334-5|Likely benign|http://loinc.org",
            "Likely Pathogenic": "LA26332-9|Likely pathogenic|http://loinc.org",
        }
    )

    zygosity: dict = field(
        default_factory=lambda: {
            "Heterozygous": "LA6706-1|Heterozygous|http://loinc.org",
            "Hemizygous": "LA6707-9|Hemizygous|http://loinc.org",
            "Homozygous": "LA6705-3|Homozygous|http://loinc.org",
            "Homoplasmic": "LA6704-6|Homoplasmic|http://loinc.org",
            "Heteroplasmic": "LA6703-8|Heteroplasmic|http://loinc.org",
            "Het": "LA6706-1|Heterozygous|http://loinc.org",
            "Heterozygote": "LA6706-1|Heterozygous|http://loinc.org",
            "Hemizygote": "LA6707-9|Hemizygous|http://loinc.org",
            "het": "LA6706-1|Heterozygous|http://loinc.org",
        }
    )
