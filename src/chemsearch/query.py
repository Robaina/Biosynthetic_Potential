from typing import List, Dict, Union
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd

# Suppress RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


def compute_similarity(smiles: str, target_mol: Chem.Mol) -> float:
    """Compute Tanimoto similarity between the given SMILES string and a target molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    fp2 = AllChem.GetMorganFingerprint(target_mol, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def query_reaction_database_by_smiles(
    df: pd.DataFrame,
    target_compounds: List[str],
    thresholds: Union[float, List[float]] = 0.8,
    only_best_hit: bool = True,
) -> pd.DataFrame:
    """
    Queries a DataFrame for chemical compounds that have a SMILES similarity to any of a list of target compounds
    above specified thresholds using RDKit. Adds the query SMILES to the results for easy comparison and keeps
    all original DataFrame columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'SMILES' and 'compound_id' columns.
        target_compounds (List[str]): List of SMILES strings of the target compounds for similarity comparison.
        thresholds (Union[float, List[float]]): Minimum Tanimoto similarity thresholds for the compounds to be included in the results.
        only_best_hit (bool): If set to True, only returns the best match with the highest similarity for each unique compound_id.

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

    all_results = []
    for target_compound, threshold in zip(target_compounds, thresholds):
        target_mol = Chem.MolFromSmiles(target_compound)
        if not target_mol:
            raise ValueError(f"Invalid target SMILES string: {target_compound}")

        # Create a temporary DataFrame to avoid SettingWithCopyWarning
        temp_df = df.copy()
        temp_df["similarity"] = temp_df["SMILES"].apply(
            lambda x: compute_similarity(x, target_mol)
        )
        temp_df = temp_df[temp_df["similarity"] >= threshold]
        temp_df["query_SMILES"] = target_compound
        temp_df.rename(columns={"SMILES": "target_SMILES"}, inplace=True)

        # Add the results to all_results list
        all_results.append(temp_df)

    # Concatenate all results
    result_df = pd.concat(all_results, ignore_index=True)

    if only_best_hit:
        # Filter to get only the best hit for each compound_id after collecting all possible hits
        result_df = result_df.loc[
            result_df.groupby("compound_id")["similarity"].idxmax()
        ].reset_index(drop=True)

    # Move 'query_SMILES' right after 'target_SMILES'
    cols = result_df.columns.tolist()
    target_index = cols.index("target_SMILES")
    query_index = cols.index("query_SMILES")
    # Reorder columns
    cols.insert(target_index + 1, cols.pop(query_index))
    result_df = result_df[cols]

    return result_df.drop_duplicates().reset_index(drop=True)


def extract_genome_hits(
    results_df: pd.DataFrame,
    genomes: Dict[str, Dict[str, List[str]]],
    taxonomy: str = None,
) -> pd.DataFrame:
    """
    Extracts and merges genomic hits based on EC numbers from a results dataframe that includes SMILES similarity data.
    Optionally adds taxonomy information if a file path is provided.

    Parameters:
        results_df (pd.DataFrame): A DataFrame containing columns 'compound_id', 'compound_name', 'target_SMILES',
                                   'query_SMILES', 'ec_numbers', 'reaction_ids', and 'similarity' from SMILES queries.
        genomes (Dict[str, Dict[str, List[str]]]): A dictionary where each key is a genome identifier and each value
                                                   is a dictionary mapping 'sequence_ID' to a list of 'EC numbers'.
        taxonomy (Optional[str]): Path to a file containing taxonomy information for genomes. If not None, the file should
                                  contain two columns: 'genome_id' and 'taxonomy'.

    Returns:
        pd.DataFrame: A DataFrame with genome identifiers, sequence IDs, EC numbers, compound names, target and query SMILES,
                      and similarity scores for each genomic hit that matches EC numbers in the results dataframe. Optionally
                      includes taxonomy information.
    """
    # Generate data for all genomes
    genome_data = [
        (genome_name, sequence_id, ec_number)
        for genome_name, sequences in genomes.items()
        for sequence_id, ec_list in sequences.items()
        for ec_number in ec_list
    ]
    genome_df = pd.DataFrame(
        genome_data, columns=["genome", "sequence_ID", "ec_number"]
    )

    # Explode results on EC numbers
    expanded_results = results_df.assign(
        ec_number=results_df["ec_numbers"].str.split(";")
    ).explode("ec_number")

    # Merge the genome data with results
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

    # If a taxonomy file path is provided, add taxonomy information
    if taxonomy is not None:
        taxonomy_df = pd.read_csv(taxonomy, sep="\t")
        taxonomy_df.columns = ["genome", "taxonomy"]  # Adjust column names as needed
        output_df = pd.merge(output_df, taxonomy_df, on="genome", how="left")

    return output_df
