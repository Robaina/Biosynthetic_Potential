from typing import List, Dict, Union
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd

# Suppress RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


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
