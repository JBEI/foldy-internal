#!/usr/bin/env bash
# -------------------------------------------------
# Prerequisites:
#   • Google Cloud SDK installed and authenticated
#   • Run from the repository root
# -------------------------------------------------

DEST_ROOT="backend/folde/data"

# -------------------------------------------------
# 1️⃣ Activity CSVs
# -------------------------------------------------
for dms in \
  KCNJ2_MOUSE_Coyote-Maestas_2022_function \
  SC6A4_HUMAN_Young_2021 \
  PTEN_HUMAN_Mighell_2018 \
  S22A1_HUMAN_Yee_2023_activity \
  KKA2_KLEPN_Melnikov_2014 \
  PPARG_HUMAN_Majithia_2016 \
  MET_HUMAN_Estevam_2023 \
  MTHR_HUMAN_Weile_2021 \
  LGK_LIPST_Klesmith_2015 \
  AMIE_PSEAE_Wrenbeck_2017 \
  PAI1_HUMAN_Huttinger_2021 \
  A4GRB6_PSEAI_Chen_2020 \
  MSH2_HUMAN_Jia_2020 \
  MLAC_ECOLI_MacRae_2023 \
  RNC_ECOLI_Weeks_2023 \
  HMDH_HUMAN_Jiang_2019 \
  CAS9_STRP1_Spencer_2017_positive; do
  gsutil cp "gs://foldedata/activity/${dms}.csv" "${DEST_ROOT}/activity/${dms}.csv"
done

# -------------------------------------------------
# 2️⃣ Naturalness CSVs (model 600m – files are directly under the bucket)
# -------------------------------------------------
for dms in \
  KCNJ2_MOUSE_Coyote-Maestas_2022_function \
  SC6A4_HUMAN_Young_2021 \
  PTEN_HUMAN_Mighell_2018 \
  S22A1_HUMAN_Yee_2023_activity \
  KKA2_KLEPN_Melnikov_2014 \
  PPARG_HUMAN_Majithia_2016 \
  MET_HUMAN_Estevam_2023 \
  MTHR_HUMAN_Weile_2021 \
  LGK_LIPST_Klesmith_2015 \
  AMIE_PSEAE_Wrenbeck_2017 \
  PAI1_HUMAN_Huttinger_2021 \
  A4GRB6_PSEAI_Chen_2020 \
  MSH2_HUMAN_Jia_2020 \
  MLAC_ECOLI_MacRae_2023 \
  RNC_ECOLI_Weeks_2023 \
  HMDH_HUMAN_Jiang_2019 \
  CAS9_STRP1_Spencer_2017_positive; do
  gsutil cp "gs://foldedata/naturalness/${dms}_naturalness_600m.csv" "${DEST_ROOT}/naturalness/${dms}_naturalness_600m.csv"
done

# -------------------------------------------------
# 3️⃣ Embeddings – models 300m and 15b (files are named <DMS>_embedding_<model>.csv)
# -------------------------------------------------
for model in 300m 600m 15b; do
  for dms in \
    KCNJ2_MOUSE_Coyote-Maestas_2022_function \
    SC6A4_HUMAN_Young_2021 \
    PTEN_HUMAN_Mighell_2018 \
    S22A1_HUMAN_Yee_2023_activity \
    KKA2_KLEPN_Melnikov_2014 \
    PPARG_HUMAN_Majithia_2016 \
    MET_HUMAN_Estevam_2023 \
    MTHR_HUMAN_Weile_2021 \
    LGK_LIPST_Klesmith_2015 \
    AMIE_PSEAE_Wrenbeck_2017 \
    PAI1_HUMAN_Huttinger_2021 \
    A4GRB6_PSEAI_Chen_2020 \
    MSH2_HUMAN_Jia_2020 \
    MLAC_ECOLI_MacRae_2023 \
    RNC_ECOLI_Weeks_2023 \
    HMDH_HUMAN_Jiang_2019 \
    CAS9_STRP1_Spencer_2017_positive; do
    # Ensure target directory exists
    mkdir -p "${DEST_ROOT}/embeddings/${model}/${dms}"
    # Copy the embedding CSV for the current model
    gsutil cp "gs://foldedata/embeddings/${dms}_embedding_${model}.csv" "${DEST_ROOT}/embeddings/${model}/${dms}/"
  done
done