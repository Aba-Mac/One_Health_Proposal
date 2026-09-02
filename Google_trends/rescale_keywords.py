"""
Rescale all Google Trends batches onto one common scale using anchor words.

This version assumes trends_keywords_with_batches.csv columns are named:
    "COUNTRY::keyword::batch_N"

and trends_anchors.csv columns are named:
    "COUNTRY::anchorword::batch_N"

Every batch contains the country's anchor term. We use the anchor to
directly compare EVERY batch against one global reference batch, rather
than first rescaling within countries and then bridging countries.

For every (country, batch):

    scale_factor =
        sum(reference anchor)
        ---------------------
        sum(current batch anchor)

Then:

    rescaled keyword = raw keyword * scale_factor

The result puts all batches onto the scale of the chosen reference batch.

IMPORTANT:
-----------
Within a country, this uses the same anchor term repeatedly, so the
comparison is a batch-scaling procedure.

Across countries, however, the anchor terms differ:

    DE/AT/CH = Mücke
    IT       = Zanzara
    ES       = Mosquito
    FR       = Moustique

Therefore, comparing batches across countries assumes that these anchor
terms represent comparable underlying search demand. This is a substantive
assumption and should be stated explicitly in any analysis/publication.

The script does NOT normalize keyword values relative to the anchor
month-by-month. It only uses the anchor to estimate a single scaling
factor for each batch.

Using the sum across the entire time series reduces sensitivity to
Google Trends' integer rounding and avoids calculating unstable
row-by-row ratios.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------------------------

ANCHORS_CSV = r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance\One_Health_Proposal_Eugenia\Google_trends\trends_anchors.csv"
KEYWORDS_CSV = r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance\One_Health_Proposal_Eugenia\Google_trends\trends_keywords.csv"

OUTPUT_CSV = r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance\One_Health_Proposal_Eugenia\Google_trends\keywords_rescaled.csv"
SCALE_CSV = r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance\One_Health_Proposal_Eugenia\Google_trends\batch_scale_factors.csv"


# ---------------------------------------------------------------------------
# REFERENCE BATCH
# ---------------------------------------------------------------------------
#
# Choose ONE batch as the global reference.
#
# This is arbitrary mathematically: choosing another reference batch
# changes the absolute scale but not the relative relationships between
# batches.
#
# DE batch 1 is used here as the reference.
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
# DERIVE COUNTRY -> ANCHOR WORD
# ---------------------------------------------------------------------------
#
# Anchor columns look like:
#
#     DE::Mücke::batch_1
#     DE::Mücke::batch_2
#     IT::Zanzara::batch_1
#
# We derive the anchor word directly from the column names.
#

country_anchor = {}

for col in anchors.columns:
    country, anchor_word, batch_tag = col.split("::")
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
    batch_num = int(batch_tag.rsplit("_", 1)[1])

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
        f"Reference anchor column not found: {reference_anchor_col}"
    )

reference_anchor_sum = anchors[reference_anchor_col].sum()

if reference_anchor_sum <= 0:
    raise ValueError(
        f"Reference anchor has non-positive sum: "
        f"{reference_anchor_col} = {reference_anchor_sum}"
    )


print(
    f"Global reference batch: "
    f"{REFERENCE_COUNTRY}, batch {REFERENCE_BATCH}"
)

print(
    f"Reference anchor: {reference_anchor_col}"
)

print(
    f"Reference anchor sum: {reference_anchor_sum:.3f}"
)


# ---------------------------------------------------------------------------
# CALCULATE ONE SCALE FACTOR FOR EVERY BATCH
# ---------------------------------------------------------------------------

batch_scale = {}

for row in batches.itertuples(index=False):

    country = row.country
    batch = row.batch
    anchor_column = row.anchor_column

    anchor_sum = anchors[anchor_column].sum()

    if anchor_sum <= 0:
        raise ValueError(
            f"Anchor sum is non-positive for "
            f"{country}, batch {batch}: "
            f"{anchor_column} = {anchor_sum}"
        )

    scale_factor = reference_anchor_sum / anchor_sum

    batch_scale[(country, batch)] = scale_factor


# ---------------------------------------------------------------------------
# SAVE SCALE FACTORS FOR INSPECTION
# ---------------------------------------------------------------------------

scale_rows = []

for (country, batch), scale_factor in batch_scale.items():

    anchor_column = anchor_col(country, batch)

    scale_rows.append(
        {
            "country": country,
            "batch": batch,
            "anchor": country_anchor[country],
            "anchor_column": anchor_column,
            "anchor_sum": anchors[anchor_column].sum(),
            "reference_anchor": reference_anchor_col,
            "reference_anchor_sum": reference_anchor_sum,
            "scale_factor": scale_factor,
        }
    )

scale_df = (
    pd.DataFrame(scale_rows)
    .sort_values(["country", "batch"])
)

scale_df.to_csv(SCALE_CSV, index=False)

print()
print("Global batch scale factors:")
print()

for row in scale_df.itertuples(index=False):
    print(
        f"  {row.country} batch {row.batch}: "
        f"{row.anchor} sum={row.anchor_sum:.3f} "
        f"→ x{row.scale_factor:.3f}"
    )

print()
print(f"Scale factors saved → {SCALE_CSV}")


# ---------------------------------------------------------------------------
# RESCALE ALL KEYWORD COLUMNS
# ---------------------------------------------------------------------------
#
# Each keyword column has the form:
#
#     COUNTRY::keyword::batch_N
#
# We identify its country and batch, look up the corresponding global
# scale factor, and multiply the raw Google Trends values by that factor.
#

rescaled = {}

for col in keywords.columns:

    parts = col.split("::")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected keyword column format: {col}"
        )

    country, keyword, batch_tag = parts

    batch = int(batch_tag.rsplit("_", 1)[1])

    key = (country, batch)

    if key not in batch_scale:
        raise ValueError(
            f"No scale factor found for {country}, batch {batch}"
        )

    rescaled[col] = (
        keywords[col] * batch_scale[key]
    )


rescaled = pd.DataFrame(
    rescaled,
    index=keywords.index,
)


# ---------------------------------------------------------------------------
# SAVE RESCALED DATA
# ---------------------------------------------------------------------------

rescaled.to_csv(OUTPUT_CSV)

print()
print(f"Rescaled keyword data saved → {OUTPUT_CSV}")
print()
print("Done.")
