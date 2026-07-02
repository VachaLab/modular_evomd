# === scales.py ===
"""
Scales: reference physicochemical tables for the 20 natural amino acids.

This module centralizes the per-residue constants used across Evo-MD, keyed by
one-letter amino acid code:

  - hydrophobicity_scales: hydrophobicity values under four named scales
    (eisenberg, kyte-doolittle, wimley-white, fauchere-pliska). The scale to
    use is selected elsewhere (e.g. by the Instructor's hydrofobic_scale).
  - aa_charges: net residue charge (+1 for R/K, -1 for D/E, 0 otherwise).
  - aa_volumes: per-residue volume (see WARNING below about units).
  - aa_masses: per-residue mass on a normalized scale (see note below).
  - aa_group: an integer chemical-group classification (signed, see legend).

The Classification class provides residue-set shortcuts (which letters are
positive, aromatic, etc.) as plain strings, for membership tests.

NOTE / DATA CAVEATS (not documentation-only; worth verifying):
  * aa_volumes mixes scales: N, C, Q are given as large values (~86-114) while
    all other residues are ~0.5-2.0. Summing/averaging volumes across residues
    will be dominated by those three. Verify the intended unit/source.
  * aa_masses are not absolute Daltons (W=2.04, G=0.75); they appear normalized
    (roughly Da/100). Treat them as relative unless confirmed otherwise.
  * aa_group and Classification are two independent taxonomies and do NOT agree
    residue-by-residue (e.g. H is 'amide' group -2 here but listed under
    Classification.polar). Do not assume they describe the same sets.
"""

import logging

logger = logging.getLogger(__name__)


class Scales:
    """
    Static lookup tables of amino acid physicochemical properties.

    All attributes are class-level dictionaries (or nested dictionaries) keyed
    by one-letter amino acid code. Intended to be used as read-only reference
    data, not instantiated.
    """

    name = 'residue'

    # Hydrophobicity by scale name -> {residue: value}. Higher generally means
    # more hydrophobic, but the numeric range and sign convention differ per
    # scale, so values are only comparable within the same scale.
    hydrophobicity_scales = {
        'eisenberg': {
            'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04, 'Q': -0.69,
            'E': -0.62, 'G': 0.16, 'H': -0.4,  'I': 0.73, 'L': 0.53, 'K': -1.1,
            'M': 0.26, 'F': 0.61, 'P': -0.07, 'S': -0.26, 'T': -0.18, 'W': 0.37,
            'Y': 0.02, 'V': 0.54,  # Proc. Natl. Acad. Sci. USA, 1984, 81, 140-144.
        },
        'kyte-doolittle': {
            'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,  'Q': -3.5,
            'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
            'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
            'Y': -1.3, 'V': 4.2,
        },
        'wimley-white': {
            'A': 0.17, 'R': 0.81, 'N': 0.42, 'D': 1.23, 'C': -0.24, 'Q': 0.58,
            'E': 2.02, 'G': 0.01, 'H': 0.96, 'I': -0.31, 'L': -0.56, 'K': 0.99,
            'M': -0.23, 'F': -1.13, 'P': 0.45, 'S': 0.13, 'T': 0.14, 'W': -1.85,
            'Y': -0.94, 'V': 0.07,
        },
        'fauchere-pliska': {
            'A': 0.31,  'R': -1.01, 'N': -0.60, 'D': -0.77, 'C': 1.54,  'Q': -0.22,
            'E': -0.64, 'G': 0.00,  'H': 0.13,  'I': 1.80,  'L': 1.70,  'K': -0.99,
            'M': 1.23,  'F': 1.79,  'P': 0.72,  'S': -0.04, 'T': 0.26,  'W': 2.25,
            'Y': 0.96,  'V': 1.22,
        },  # Eur. J. Med. Chem. - Chim. Ther., 1983, 18(4), 369-375.
    }

    # Net residue charge: +1 for R/K, -1 for D/E, 0 for all others.
    aa_charges = {
        'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0,
        'G': 0.0, 'H': 0.0, 'I': 0.0, 'L': 0.0,
        'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0,
        'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
        'R': 1.0, 'K': 1.0,
        'D': -1.0, 'E': -1.0,
    }

    # Per-residue side-chain/residue volume in cubic angstroms (Å³).
    # Source: Zamyatnin (1972), standard amino acid residue volumes.
    aa_volumes = {
        'A': 88.6,  'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
        'Q': 143.8, 'E': 138.4, 'G': 60.1,  'H': 153.2, 'I': 166.7,
        'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
        'S': 89.0,  'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
    }

    # Per-residue monoisotopic mass in daltons (Da), i.e. amino acid minus
    # one water molecule (the residue mass inside a peptide chain).
    # Sum of residues + 18.02 (one H2O) gives the peptide's mass.
    aa_masses = {
        'A': 71.08,  'R': 156.19, 'N': 114.10, 'D': 115.09, 'C': 103.14,
        'Q': 128.13, 'E': 129.12, 'G': 57.05,  'H': 137.14, 'I': 113.16,
        'L': 113.16, 'K': 128.17, 'M': 131.19, 'F': 147.18, 'P': 97.12,
        'S': 87.08,  'T': 101.10, 'W': 186.21, 'Y': 163.18, 'V': 99.13,
    }

    # Arbitrary chemical-group classification, encoded as a signed integer:
    #   -3: negative (acidic)   -2: amides         -1: polar (hydroxyl)
    #    0: non-polar (sulfur)
    #    1: aliphatic           2: aromatic         3: positive (basic)
    # Note: this grouping is independent from the Classification sets below and
    # does not necessarily agree with them residue-by-residue.
    aa_group = {
        'D': -3, 'E': -3,
        'N': -2, 'Q': -2, 'H': -2, 
        'S': -1, 'T': -1,
        'C': 0, 'M': 0, 
        'A': 1, 'G': 1, 'I': 1, 'L': 1, 'P': 1, 'V': 1,
        'W': 2, 'Y': 2,  'F': 2,
        'R': 3, 'K': 3,
    }


class Classification:
    """
    Residue-set shortcuts as one-letter-code strings.

    Each attribute lists the residues belonging to a chemical category, for use
    in membership tests (e.g. `if residue in Classification.positive`). These
    sets are independent from Scales.aa_group and need not match it.

    NOTE: `aliphtic` is spelled without the second 'a' (kept as-is for
    backwards compatibility with any code referencing it).
    """

    positive = 'KR'      # basic / positively charged
    negative = 'DE'      # acidic / negatively charged
    thiol = 'C'          # cysteine (thiol side chain)
    amide = 'NQ'         # asparagine, glutamine
    alcohol = 'ST'       # serine, threonine (hydroxyl)
    aromatic = 'FWY'     # aromatic ring side chains
    aliphatic = 'AVLI'    # aliphatic (note the misspelling of 'aliphatic')
    thioether = 'M'      # methionine
    polar = 'H'          # histidine (treated as polar here)
    nonpolar = 'GP'      # glycine, proline

    # One-letter code -> stable integer id (e.g. for encoding sequences numerically).
    str2int = {
        'A': 1, 'N': 2, 'C': 3, 'Q': 4,
        'G': 5, 'H': 6, 'I': 7, 'L': 8,
        'M': 9, 'F': 10, 'P': 11, 'S': 12,
        'T': 13, 'W': 14, 'Y': 15, 'V': 16,
        'R': 17, 'K': 18,
        'D': 19, 'E': 20,
    }


if __name__ == '__main__':
    pass
