from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from rdkit import Chem


def preprocess_reactions(
    reactions_path: Path, compounds_path: Path, complete_smiles: bool = True
) -> list[dict]:
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


def load_reactions(file_path: Path) -> list[dict]:
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


def load_compounds(file_path: Path) -> dict[str, dict]:
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


def reaction_has_complete_SMILES(reaction: dict) -> bool:
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
