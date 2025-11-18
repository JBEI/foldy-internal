import app.helpers.e1_client  # triggers monkey patch
import numpy as np
import pandas as pd
from app.helpers.esm_client import FoldyESMClient

seq = "MKTAYIAKQR"  # or notebook short
print("Creating ESM2 client...")
esm = FoldyESMClient.get_client("esm2_t6_8M_UR50D")
print("ESM2 client created successfully")
e1 = FoldyESMClient.get_client("e1-300m")  # via monkeypatch
print("Running ESM2 embed...")
esm_emb = esm.embed(seq)
print("ESM2 embed completed")
e1_emb = e1.embed(seq)
print("Running ESM2 logits...")
esm_log = esm.get_logits(seq)
print("ESM2 logits completed")
print("Creating E1 client...")
e1 = FoldyESMClient.get_client("e1-300m")  # via monkeypatch
print("E1 client created successfully")
print("Running E1 embed...")
e1_emb = e1.embed(seq)
print("E1 embed completed")
print("Running E1 logits...")
e1_log = e1.get_logits(seq)
print("E1 logits completed")
print("ESM embed:", type(esm_emb), len(esm_emb), len(esm_emb[0]))
print("E1 embed:", type(e1_emb), len(e1_emb), len(e1_emb[0]))
print("ESM logits shape:", esm_log.shape, esm_log.columns.tolist())
print("E1 logits shape:", e1_log.shape, e1_log.columns.tolist())
print("Sample ESM seq_id:", esm_log["seq_id"].head().tolist())
print("Sample E1 seq_id:", e1_log["seq_id"].head().tolist())
np.save("esm_embeds.npy", np.array(esm_emb))
np.save("e1_embeds.npy", np.array(e1_emb))
esm_log.to_pickle("esm_logits.pkl")
e1_log.to_pickle("e1_logits.pkl")
assert esm_log.columns.equals(e1_log.columns), "Col mismatch"
assert len(esm_log) == len(e1_log), "Row mismatch"
print("Structure verified: cols/rows match.")
