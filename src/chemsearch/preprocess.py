import re
import os
import csv
from typing import List, Dict, Any
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


def preprocess_reactions(
    reactions_path: Path, compounds_path: Path, complete_smiles: bool = True
) -> List[Dict]:
    """
    Preprocesses a list of reaction dictionaries by assigning compounds to them,
    optionally filtering out reactions without complete SMILES information.

    This function first loads a list of reactions from a specified path. It then
    enriches these reactions with compounds information from a specified compounds database.
    If the complete_smiles flag is set, the function filters the reactions to include only
    those with complete SMILES information.

    Args:
        reactions_path (Path): Path to the JSON or CSV file containing the reactions database.
        compounds_path (Path): Path to the JSON or CSV file containing the compounds database.
        complete_smiles (bool): Flag to indicate whether to filter reactions based on the
                                completeness of SMILES strings.

    Returns:
        List[Dict]: A list of processed reaction dictionaries, each potentially containing
                    detailed compound information and filtered based on SMILES completeness.
    """
    reactions = load_reactions(reactions_path)
    reactions = assign_compounds_to_modelseed_reactions(reactions, compounds_path)

    if complete_smiles:
        reactions = [rxn for rxn in reactions if reaction_has_complete_SMILES(rxn)]

    return reactions


def load_reactions(file_path: Path) -> List[Dict]:
    """
    Loads the reactions from a JSON file into a list.

    Args:
    file_path (Path): A Path object pointing to the JSON file containing the reaction database.

    Returns:
    list[dict]: A list of dictionaries, each representing a reaction.
    """
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as file:
        reactions = json.load(file)
    return reactions


def load_compounds(file_path: Path) -> Dict[str, Dict]:
    """
    Loads the compounds from a JSON file into a dictionary.

    Args:
    file_path (Path): A Path object pointing to the JSON file containing the compounds database.

    Returns:
    dict[str, dict]: A dictionary with compound IDs as keys and compound information as values.
    """
    with file_path.open("r", encoding="utf-8") as file:
        compounds = json.load(file)
    return {compound["id"]: compound for compound in compounds}


def assign_compounds_to_modelseed_reactions(
    reactions: list, compounds_path: Path
) -> list:
    """
    Extracts compound IDs from reaction objects, classifies them as substrates or products,
    and adds additional compound details from a compounds database. The original keys in each
    reaction dictionary are retained, and two new keys, substrates and products, are added.

    Args:
        reactions (list[dict]): A list of reaction dictionaries.
        compounds_path (Path): Path to the JSON file containing the compounds database.

    Returns:
        list[dict]: A list of dictionaries, each retaining their original data and including
                    substrates and products with additional compound details and stoichiometry.
    """
    compound_db = load_compounds(compounds_path)
    for reaction in reactions:
        stoichiometry = reaction.get("stoichiometry", "")
        substrates = []
        products = []
        for compound in stoichiometry.split(";"):
            parts = compound.split(":")
            if len(parts) >= 2:
                coefficient, compound_id = parts[0], parts[1]
                try:
                    coef = float(coefficient)
                    compound_info = compound_db.get(compound_id, {})
                    compound_detail = {
                        "id": compound_id,
                        "abbreviation": compound_info.get("abbreviation", ""),
                        "smiles": compound_info.get("smiles", ""),
                        "inchikey": compound_info.get("inchikey", ""),
                        "formula": compound_info.get("formula", ""),
                        "stoichiometry": coef,
                    }
                    if coef < 0:
                        substrates.append(compound_detail)
                    elif coef > 0:
                        products.append(compound_detail)
                except ValueError:
                    pass  # Skip invalid coefficients
        reaction["substrates"] = substrates
        reaction["products"] = products
    return reactions


def reaction_has_complete_SMILES(reaction: Dict) -> bool:
    """
    Checks if a reaction dictionary contains valid SMILES for all its substrates and products.

    Args:
    reaction (dict): A reaction dictionary with 'substrates' and 'products' keys.

    Returns:
    bool: True if the reaction has valid SMILES for all substrates and products, False otherwise.
    """
    for compound_list in [reaction.get("substrates", []), reaction.get("products", [])]:
        for compound in compound_list:
            smiles = compound.get("smiles", "")
            if (not smiles) or (Chem.MolFromSmiles(smiles) is None):
                return False
    return True


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
    Processes eggNOG annotation TSV files in the specified input directory, extracting specific columns from each file,
    cleaning the data, and saving the results to a new TSV file in the specified output directory.
    The function assumes that relevant data starts from the fifth row and that the files have
    '.annotations.tsv' in their names, which is replaced with '.tsv' in the output.
    It extracts the genome ID from the first part of the '#query' names, until the fifth "_".

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
                # Process to extract genome_id and rename columns
                df["genome_id"] = df["#query"].apply(
                    lambda x: "_".join(x.split("_")[:5])
                )
                df["protein_id"] = df["#query"]
                cleaned_df = df[["genome_id", "protein_id", "EC"]].iloc[:-3, :]
                cleaned_df.columns = ["genome_id", "protein_id", "EC"]

                # Handling EC numbers that are dashes or empty
                cleaned_df["EC"] = cleaned_df["EC"].apply(
                    lambda x: "" if x in ["-", ""] else x
                )

                output_filename = filename.replace(".annotations.tsv", ".tsv")
                output_file_path = os.path.join(output_directory, output_filename)
                cleaned_df.to_csv(output_file_path, sep="\t", index=False)


def parse_genome_ec_numbers(
    directory_path: str,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Parses multiple TSV files from a given directory, each containing genome IDs, protein sequence IDs and their associated EC numbers.
    Returns a dictionary where the outer keys correspond to genome IDs, and each inner dictionary maps protein IDs to lists of EC numbers.
    Entries with a dash "-" or empty string as the EC number are treated as having no EC numbers.

    Args:
        directory_path (str): The path to the directory containing TSV files.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary mapping genome IDs to dictionaries of protein IDs and lists of EC numbers.
    """
    genomes = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".tsv"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r", newline="") as file:
                tsv_reader = csv.reader(file, delimiter="\t")
                next(tsv_reader)  # Skip the header row
                for row in tsv_reader:
                    genome_id = row[0]
                    protein_id = row[1]
                    ec_numbers = row[2].strip()
                    if ec_numbers == "-" or not ec_numbers:
                        ec_list = []
                    else:
                        ec_list = ec_numbers.split(",")

                    if genome_id not in genomes:
                        genomes[genome_id] = {}
                    genomes[genome_id][protein_id] = ec_list

    return genomes
