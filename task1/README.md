# Task 1: Social Group Discovery in Trending Events

## Dataset Download

[Download Task 1 Dataset.zip](https://github.com/icdm2026ARRLChallenge/icdm2026ARRLChallenge.github.io/releases/download/datasets-v1.0/Task1-Dataset.zip)

The dataset is distributed through GitHub Releases. Extract the archive to obtain the `Dataset/` directory.

## Embedding Encoder

The provided embeddings are generated using **Qwen3-VL-Embedding-2B**, with **2,048 dimensions** per vector. This applies to the feature vectors in `user_name.pt`, `user_description.pt`, `user_avatar.pt` and `post_text.pt`.

## Feature Alignment and Missing Values

Embedding matrices have shape `[N, 2048]`; rows correspond to the identifiers stored in the same file. Align features by identifier rather than assuming the same row order across files. In `user_name.pt`, `user_description.pt` and `user_avatar.pt`, `mask=false` indicates a missing feature, represented by a zero vector. Use the mask to distinguish missing features from available embeddings. Missing `followers_count` or `following_count` values in `users.pt` are stored as `None`; these indicate missing data, not zero counts. Handle missing values before numeric conversion or normalization.
