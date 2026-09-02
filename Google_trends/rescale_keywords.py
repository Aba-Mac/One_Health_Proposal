"""
Rescale all Google Trends batches onto one common scale using anchor words,
then aggregate the rescaled keywords by country and Sub-type.

INPUTS
------
1. trends_anchors.csv
   Anchor columns:
       COUNTRY::anchorword::batch_N

2. trends_keywords.csv
   Google Trends keyword columns:
       COUNTRY::keyword::batch_N

3. Words_dataset.csv
   Keyword metadata, including:
       Keyword
       Sub-type
       System
       Type

PROCESS
-------
For every (country, batch):

    scale_factor =
        sum(reference anchor)
        ---------------------
        sum(current batch anchor)

Every keyword in that country/batch is multiplied by this factor.

Then the rescaled keyword data are aggregated by:

    COUNTRY × Sub-type

System is deliberately ignored.

OUTPUTS
-------
1. keywords_rescaled.csv
   All individual keyword time series after batch rescaling.

2. batch_scale_factors.csv
   Scale factor used for every country/batch.

3. country_subtype_sums.csv
   Time series summed across all keywords belonging to each
   country/Sub-type combination.

IMPORTANT ASSUMPTION
--------------------
Within a country, the same anchor term is used across batches.

Across countries, the anchors differ:

    DE/AT/CH = Mücke
    IT       = Zanzara
    ES       = Mosquito
    FR       = Moustique

Therefore, comparing batches across countries assumes that these
different anchor terms represent comparable underlying search demand.

The anchor is used only to calculate one scaling factor per batch.
There is no month-by-month normalization relative to the anchor.

The aggregation step sums the RESCALED keyword values. It does not
aggregate raw Google Trends values.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------------------------

ANCHORS_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Google_trends\trends_anchors.csv"
)

KEYWORDS_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Google_trends\trends_keywords.csv"
)

# Keyword metadata file containing Sub-type
METADATA_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Datasets\Keywords.csv"
)

# Encoding of Words_dataset.csv
METADATA_ENCODING = "cp1252"

# Existing output
OUTPUT_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Google_trends\keywords_rescaled.csv"
)

SCALE_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Google_trends\batch_scale_factors.csv"
)

SUBTYPE_SUM_CSV = (
    r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance"
    r"\One_Health_Proposal_Eugenia\Google_trends"
    r"\country_subtype_sums.csv"
)


# ---------------------------------------------------------------------------
# REFERENCE BATCH
# ---------------------------------------------------------------------------
#
# Choose ONE batch as the global reference.
#
# Changing the reference batch changes the absolute scale, but not the
# relative relationships between batches.
#

REFERENCE_COUNTRY = "DE"
REFERENCE_BATCH = 1


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

anchors = (
    pd.read_csv(ANCHORS_CSV)
    .set_index("time [UTC]")
)

keywords = (
    pd.read_csv(KEYWORDS_CSV)
    .set_index("time [UTC]")
)


# ---------------------------------------------------------------------------
# LOAD KEYWORD METADATA
# ---------------------------------------------------------------------------
#
# Words_dataset.csv contains:
#
#   Keyword
#   Language
#   System
#   Sub-type
#   Type
#
# We only need:
#
#   Keyword -> Sub-type
#
# System is deliberately ignored.
#

kw_meta = pd.read_csv(
    METADATA_CSV,
    encoding=METADATA_ENCODING
)

kw_meta.columns = kw_meta.columns.str.strip()

for col in ["Keyword", "Language", "System", "Sub-type", "Type"]:
    if col in kw_meta.columns:
        kw_meta[col] = (
            kw_meta[col]
            .astype(str)
            .str.strip()
        )

required_columns = {"Keyword", "Sub-type"}

missing = required_columns - set(kw_meta.columns)

if missing:
    raise ValueError(
        f"Metadata file is missing required columns: {missing}"
    )


# First occurrence wins for duplicate keywords
meta = (
    kw_meta
    .drop_duplicates(subset="Keyword")
    .set_index("Keyword")[["Sub-type"]]
    .to_dict(orient="index")
)

print(
    f"Loaded metadata for {len(meta)} unique keywords"
)

print(
    "Sub-types:",
    sorted({
        v["Sub-type"]
        for v in meta.values()
    })
)


# ---------------------------------------------------------------------------
# DERIVE COUNTRY -> ANCHOR WORD
# ---------------------------------------------------------------------------
#
# Anchor columns look like:
#
#     DE::Mücke::batch_1
#     DE::Mücke::batch_2
#     IT::Zanzara::batch_1
#

country_anchor = {}

for col in anchors.columns:

    parts = col.split("::")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected anchor column format: {col}"
        )

    country, anchor_word, batch_tag = parts

    country_anchor[country] = anchor_word


def anchor_col(country, batch_num):
    """Return the anchor-column name for a country and batch."""

    return (
        f"{country}::"
        f"{country_anchor[country]}::"
        f"batch_{batch_num}"
    )


# ---------------------------------------------------------------------------
# IDENTIFY ALL BATCHES
# ---------------------------------------------------------------------------

all_batches = []

for col in anchors.columns:

    country, anchor_word, batch_tag = col.split("::")

    batch_num = int(
        batch_tag.rsplit("_", 1)[1]
    )

    all_batches.append(
        {
            "country": country,
            "batch": batch_num,
            "anchor_column": col,
        }
    )

batches = pd.DataFrame(all_batches)


# ---------------------------------------------------------------------------
# REFERENCE ANCHOR
# ---------------------------------------------------------------------------

reference_anchor_col = anchor_col(
    REFERENCE_COUNTRY,
    REFERENCE_BATCH,
)

if reference_anchor_col not in anchors.columns:
    raise ValueError(
        f"Reference anchor column not found: "
        f"{reference_anchor_col}"
    )

reference_anchor_sum = anchors[
    reference_anchor_col
].sum()

if reference_anchor_sum <= 0:
    raise ValueError(
        f"Reference anchor has non-positive sum: "
        f"{reference_anchor_col} = "
        f"{reference_anchor_sum}"
    )


print()
print(
    f"Global reference batch: "
    f"{REFERENCE_COUNTRY}, "
    f"batch {REFERENCE_BATCH}"
)

print(
    f"Reference anchor: "
    f"{reference_anchor_col}"
)

print(
    f"Reference anchor sum: "
    f"{reference_anchor_sum:.3f}"
)


# ---------------------------------------------------------------------------
# CALCULATE ONE SCALE FACTOR FOR EVERY BATCH
# ---------------------------------------------------------------------------

batch_scale = {}

for row in batches.itertuples(index=False):

    country = row.country
    batch = row.batch
    anchor_column = row.anchor_column

    anchor_sum = anchors[
        anchor_column
    ].sum()

    if anchor_sum <= 0:
        raise ValueError(
            f"Anchor sum is non-positive for "
            f"{country}, batch {batch}: "
            f"{anchor_column} = {anchor_sum}"
        )

    scale_factor = (
        reference_anchor_sum /
        anchor_sum
    )

    batch_scale[
        (country, batch)
    ] = scale_factor


# ---------------------------------------------------------------------------
# SAVE SCALE FACTORS
# ---------------------------------------------------------------------------

scale_rows = []

for (
    country,
    batch
), scale_factor in batch_scale.items():

    anchor_column = anchor_col(
        country,
        batch
    )

    scale_rows.append(
        {
            "country": country,
            "batch": batch,
            "anchor": country_anchor[country],
            "anchor_column": anchor_column,
            "anchor_sum": anchors[
                anchor_column
            ].sum(),
            "reference_anchor": reference_anchor_col,
            "reference_anchor_sum": (
                reference_anchor_sum
            ),
            "scale_factor": scale_factor,
        }
    )

scale_df = (
    pd.DataFrame(scale_rows)
    .sort_values(
        ["country", "batch"]
    )
)

scale_df.to_csv(
    SCALE_CSV,
    index=False
)

print()
print("Global batch scale factors:")
print()

for row in scale_df.itertuples(
    index=False
):

    print(
        f"  {row.country} batch {row.batch}: "
        f"{row.anchor} "
        f"sum={row.anchor_sum:.3f} "
        f"→ x{row.scale_factor:.3f}"
    )

print()
print(
    f"Scale factors saved → {SCALE_CSV}"
)


# ---------------------------------------------------------------------------
# RESCALE ALL KEYWORD COLUMNS
# ---------------------------------------------------------------------------
#
# Keyword columns look like:
#
#     DE::keyword::batch_1
#
# We identify the country and batch, then apply the corresponding
# global scale factor.
#

rescaled = {}

for col in keywords.columns:

    parts = col.split("::")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected keyword column format: {col}"
        )

    country, keyword, batch_tag = parts

    batch = int(
        batch_tag.rsplit("_", 1)[1]
    )

    key = (country, batch)

    if key not in batch_scale:
        raise ValueError(
            f"No scale factor found for "
            f"{country}, batch {batch}"
        )

    rescaled[col] = (
        keywords[col] *
        batch_scale[key]
    )


rescaled = pd.DataFrame(
    rescaled,
    index=keywords.index,
)


# ---------------------------------------------------------------------------
# SAVE RESCALED KEYWORD DATA
# ---------------------------------------------------------------------------

rescaled.to_csv(
    OUTPUT_CSV
)

print()
print(
    f"Rescaled keyword data saved → "
    f"{OUTPUT_CSV}"
)


# ---------------------------------------------------------------------------
# AGGREGATE RESCALED VALUES BY COUNTRY × SUB-TYPE
# ---------------------------------------------------------------------------
#
# Each rescaled column is:
#
#     COUNTRY::keyword::batch_N
#
# We:
#
#   1. extract country
#   2. extract keyword
#   3. look up its Sub-type
#   4. add the keyword's rescaled time series to the
#      corresponding COUNTRY × Sub-type group
#
# System is NOT used.
#
# For example:
#
#     DE::keyword_A::batch_1 -> DE::Subtype_X
#     DE::keyword_B::batch_1 -> DE::Subtype_X
#     DE::keyword_C::batch_2 -> DE::Subtype_X
#
# become:
#
#     DE::Subtype_X
#
# where each timestamp is:
#
#     keyword_A + keyword_B + keyword_C
#
# after batch rescaling.
#

subtype_groups = {}

missing_metadata = []
unknown_subtypes = set()

for col in rescaled.columns:

    parts = col.split("::")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected rescaled column format: {col}"
        )

    country, keyword, batch_tag = parts

    # Look up keyword metadata
    if keyword not in meta:

        missing_metadata.append(
            {
                "column": col,
                "country": country,
                "keyword": keyword,
            }
        )

        continue

    subtype = meta[keyword]["Sub-type"]

    if pd.isna(subtype) or subtype == "":
        unknown_subtypes.add(
            keyword
        )
        continue

    # Define the aggregation column
    group_name = (
        f"{country}::{subtype}"
    )

    if group_name not in subtype_groups:

        subtype_groups[group_name] = (
            rescaled[col].copy()
        )

    else:

        subtype_groups[group_name] = (
            subtype_groups[group_name]
            .add(
                rescaled[col],
                fill_value=0
            )
        )


# ---------------------------------------------------------------------------
# REPORT MISSING METADATA
# ---------------------------------------------------------------------------

if missing_metadata:

    print()
    print(
        "WARNING: The following keywords were "
        "not found in Words_dataset.csv:"
    )

    for item in missing_metadata:
        print(
            f"  {item['country']} :: "
            f"{item['keyword']} "
            f"({item['column']})"
        )

    print(
        f"\nTotal missing keywords: "
        f"{len(missing_metadata)}"
    )

    print(
        "These keywords were NOT included in "
        "the country/Sub-type aggregation."
    )


if unknown_subtypes:

    print()
    print(
        "WARNING: The following keywords have "
        "no valid Sub-type:"
    )

    for keyword in sorted(
        unknown_subtypes
    ):
        print(
            f"  {keyword}"
        )

    print(
        "These keywords were NOT included in "
        "the country/Sub-type aggregation."
    )


# ---------------------------------------------------------------------------
# CONVERT AGGREGATION TO DATAFRAME
# ---------------------------------------------------------------------------

country_subtype_sums = pd.DataFrame(
    subtype_groups,
    index=rescaled.index,
)

# Sort columns alphabetically
country_subtype_sums = (
    country_subtype_sums
    .sort_index(axis=1)
)


# ---------------------------------------------------------------------------
# SAVE COUNTRY × SUB-TYPE DATA
# ---------------------------------------------------------------------------

country_subtype_sums.to_csv(
    SUBTYPE_SUM_CSV
)

print()
print(
    f"Country/Sub-type aggregation saved → "
    f"{SUBTYPE_SUM_CSV}"
)

print()
print(
    "Aggregated columns:"
)

for col in country_subtype_sums.columns:
    print(
        f"  {col}"
    )


# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

print()
print("Done.")