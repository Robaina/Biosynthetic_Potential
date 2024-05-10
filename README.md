# Search genomes for chemical synthetic potential with chemsearch

Design a chemical potential metric for MAGs:

1) enter a compound of interest SMILES (drug with high commercial value)
2) search for compounds associated to each MAG (via EC, reactions, modelSEED) which are similar up to a threshold (RDKit),
3) This means each MAG's metabolic network is able to produce structuraly similar compounds
4) Possible extension: look for biosynthetic modules associated to each compound

## Annotate genomes with EC numbers

1. [DeepProZyme](https://github.com/kaistsystemsbiology/DeepProZyme)