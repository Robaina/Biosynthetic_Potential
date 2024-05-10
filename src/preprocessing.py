import re
import os
import csv
from typing import List, Dict, Any, Union
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd

# Suppress RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


def process_reactions_to_dataframe(reactions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Processes a list of chemical reactions to extract compound details and creates a pandas DataFrame.

    Handles missing reaction abbreviations and cleans up EC numbers and reaction ID formatting issues.

    Parameters:
        reactions (List[Dict[str, Any]]): A list of reaction dictionaries, each potentially containing:
            - 'abbreviation': the reaction abbreviation (could be missing or empty)
            - 'ec_numbers': a list of EC numbers associated with the reaction (may be None)
            - 'substrates': a list of compound dictionaries
            - 'products': a list of compound dictionaries

    Returns:
        pd.DataFrame: A DataFrame with columns ['compound_id', 'compound_name', 'SMILES', 'ec_numbers', 'reaction_ids'],
                      filtered to remove entries with empty EC numbers and duplicates.
    """
    compound_details = {}

    def process_compounds(
        compounds: List[Dict[str, str]], ec_numbers: str, reaction_id: str
    ) -> None:
        for compound in compounds:
            compound_id = compound["id"]
            if compound_id not in compound_details:
                compound_details[compound_id] = {
                    "compound_name": compound.get("abbreviation", ""),
                    "SMILES": compound.get("smiles", ""),
                    "ec_numbers": set(),
                    "reaction_ids": set(),
                }
            cleaned_ec_numbers = [
                re.sub(r"^EC-|^;", "", ec).strip()
                for ec in ec_numbers.split(";")
                if ec.strip()
            ]
            if cleaned_ec_numbers:  # Only update if there are valid entries
                compound_details[compound_id]["ec_numbers"].update(cleaned_ec_numbers)
            if reaction_id:
                compound_details[compound_id]["reaction_ids"].add(reaction_id)

    for reaction in reactions:
        reaction_id = reaction.get("abbreviation", "No Abbreviation Provided")
        # Ensure ec_numbers is not None
        ec_numbers_list = reaction.get("ec_numbers")
        if ec_numbers_list is None:
            ec_numbers_list = []
        ec_numbers = ";".join([ec for ec in ec_numbers_list if ec])
        process_compounds(reaction.get("substrates", []), ec_numbers, reaction_id)
        process_compounds(reaction.get("products", []), ec_numbers, reaction_id)

    data = {
        "compound_id": [],
        "compound_name": [],
        "SMILES": [],
        "ec_numbers": [],
        "reaction_ids": [],
    }

    for compound_id, details in compound_details.items():
        data["compound_id"].append(compound_id)
        data["compound_name"].append(details["compound_name"])
        data["SMILES"].append(details["SMILES"])
        data["ec_numbers"].append(";".join(details["ec_numbers"]).strip(";"))
        data["reaction_ids"].append(";".join(details["reaction_ids"]).strip(";"))

    df = pd.DataFrame(data)
    df = df[df["ec_numbers"] != ""].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def process_eggnog_annotation_files(
    input_directory: str, output_directory: str
) -> None:
    """
    Processes eggNOG annotatiion TSV files in the specified input directory, extracting specific columns from each file,
    cleaning the data, and saving the results to a new TSV file in the specified output directory.
    The function assumes that relevant data starts from the fifth row and that the files have
    '.annotations.tsv' in their names, which is replaced with '.tsv' in the output.

    Parameters:
        input_directory (str): Path to the directory containing the input TSV files.
        output_directory (str): Path to the directory where the output TSV files will be stored.
    """
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".annotations.tsv"):
            input_file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(input_file_path, sep="\t", header=4)
            if "#query" in df.columns and "EC" in df.columns:
                cleaned_df = df[["#query", "EC"]].iloc[:-3, :]
                output_filename = filename.replace(".annotations.tsv", ".tsv")
                output_file_path = os.path.join(output_directory, output_filename)
                cleaned_df.to_csv(output_file_path, sep="\t", index=False)


def query_reaction_database_by_smiles(
    df: pd.DataFrame,
    target_compounds: List[str],
    thresholds: Union[float, List[float]] = 0.8,
) -> pd.DataFrame:
    """
    Queries a DataFrame for chemical compounds that have a SMILES similarity to any of a list of target compounds
    above specified thresholds using RDKit. Adds the query SMILES to the results for easy comparison and keeps
    all original DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'SMILES' column with compound SMILES strings.
        target_compounds (List[str]): List of SMILES strings of the target compounds for similarity comparison.
        thresholds (Union[float, List[float]]): Minimum Tanimoto similarity thresholds for the compounds to be included in the results.
                                                If a single float is provided, it applies to all target compounds.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only compounds with similarity above the respective thresholds for any target compound,
                      with columns for both the query and target SMILES strings next to each other, and all other original columns retained.
    """
    # Validate and prepare thresholds
    if isinstance(thresholds, float):
        thresholds = [thresholds] * len(target_compounds)
    elif len(thresholds) != len(target_compounds):
        raise ValueError(
            "Thresholds list must match the length of target compounds list if provided as a list."
        )

    result_df = pd.DataFrame()
    for target_compound, threshold in zip(target_compounds, thresholds):
        target_mol = Chem.MolFromSmiles(target_compound)
        if not target_mol:
            raise ValueError(f"Invalid target SMILES string: {target_compound}")

        df["similarity"] = df["SMILES"].apply(
            lambda x: compute_similarity(x, target_mol)
        )
        filtered_df = df[df["similarity"] >= threshold]
        filtered_df = filtered_df.copy()
        filtered_df.rename(columns={"SMILES": "target_SMILES"}, inplace=True)
        filtered_df["query_SMILES"] = target_compound
        # Reorder columns to ensure 'query_SMILES' follows 'target_SMILES'
        cols = filtered_df.columns.tolist()
        target_index = cols.index("target_SMILES")
        # Move 'query_SMILES' right after 'target_SMILES'
        cols.insert(target_index + 1, cols.pop(cols.index("query_SMILES")))
        filtered_df = filtered_df[cols]
        result_df = pd.concat([result_df, filtered_df], ignore_index=True)

    result_df = result_df.drop_duplicates().reset_index(drop=True)

    return result_df


def compute_similarity(smiles: str, target_mol: Chem.Mol) -> float:
    """
    Compute Tanimoto similarity between a SMILES string and a target RDKit Mol object.

    Args:
    smiles (str): SMILES string to convert to Mol and compare.
    target_mol (Chem.Mol): RDKit Mol object of the target compound.

    Returns:
    float: Tanimoto similarity score.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0
    fp_mol = AllChem.GetMorganFingerprint(mol, 2)
    fp_target = AllChem.GetMorganFingerprint(target_mol, 2)
    return DataStructs.TanimotoSimilarity(fp_mol, fp_target)


def extract_genome_hits(
    results_df: pd.DataFrame, genomes: Dict[str, Dict[str, List[str]]]
) -> pd.DataFrame:
    """
    Extracts and merges genomic hits based on EC numbers from a results dataframe that includes SMILES similarity data.

    Parameters:
        results_df (pd.DataFrame): A DataFrame containing columns 'compound_id', 'compound_name', 'target_SMILES',
                                   'query_SMILES', 'ec_numbers', 'reaction_ids', and 'similarity' from SMILES queries.
        genomes (Dict[str, Dict[str, List[str]]]): A dictionary where each key is a genome identifier and each value
                                                   is a dictionary mapping 'sequence_ID' to a list of 'EC numbers'.

    Returns:
        pd.DataFrame: A DataFrame with genome identifiers, sequence IDs, EC numbers, compound names, target and query SMILES,
                      and similarity scores for each genomic hit that matches EC numbers in the results dataframe.
    """
    genome_data = [
        (genome_name, sequence_id, ec_number)
        for genome_name, sequences in genomes.items()
        for sequence_id, ec_list in sequences.items()
        for ec_number in ec_list
    ]
    genome_df = pd.DataFrame(
        genome_data, columns=["genome", "sequence_ID", "ec_number"]
    )

    # Handle the list of EC numbers, assuming ec_numbers column contains semicolon-separated values
    expanded_results = results_df.assign(
        ec_number=results_df["ec_numbers"].str.split(";")
    ).explode("ec_number")

    # Perform the merge on ec_number
    output_df = pd.merge(genome_df, expanded_results, on="ec_number", how="inner")
    output_df = output_df[
        [
            "genome",
            "sequence_ID",
            "ec_number",
            "compound_name",
            "target_SMILES",
            "query_SMILES",
            "similarity",
        ]
    ]
    return output_df


def parse_genome_ec_numbers(
    directory_path: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Parses multiple TSV files from a given directory, each containing protein sequence IDs and their associated EC numbers.
    Returns a dictionary of dictionaries where the outer keys correspond to file names (without extensions), and
    each inner dictionary maps protein IDs to lists of EC numbers. Entries with a dash "-" or empty string as the EC number
    are treated as having no EC numbers.

    Args:
    directory_path (str): The path to the directory containing TSV files.

    Returns:
    Dict[str, Dict[str, List[str]]]: A dictionary mapping filenames to dictionaries of protein IDs and lists of EC numbers.
    """
    genomes = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".tsv"):
            filepath = os.path.join(directory_path, filename)
            ec_dict = {}
            with open(filepath, "r", newline="") as file:
                tsv_reader = csv.reader(file, delimiter="\t")
                next(tsv_reader)  # Skip the header row
                for row in tsv_reader:
                    protein_id = row[0]
                    ec_numbers = row[1].strip()
                    if ec_numbers == "-" or not ec_numbers:
                        ec_dict[protein_id] = []
                    else:
                        ec_dict[protein_id] = ec_numbers.split(",")
            file_base_name = os.path.splitext(filename)[0]
            genomes[file_base_name] = ec_dict
    return genomes
