# Data provenance

The training and evaluation data in this project are **synthetically generated**
by `src/lc_calc/generate.py` from a fixed seed. They are original and contain no
third-party content, which is also why the evaluation set is guaranteed free of
contamination from any model's pretraining corpus.

If you add a secondary evaluation on real State Examinations Commission (SEC) past
papers, do NOT commit SEC exam content or marking schemes to this repository — they
are SEC copyright. Reference them by year/paper/question instead and load locally.
