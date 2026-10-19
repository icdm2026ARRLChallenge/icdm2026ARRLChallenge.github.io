# Task 1: Social Group Discovery in Trending Events

## Dataset Download

[Download Task 1 Dataset.zip](https://github.com/icdm2026ARRLChallenge/icdm2026ARRLChallenge.github.io/releases/download/datasets-v1.0/Task1-Dataset.zip)

The dataset is distributed through GitHub Releases. Extract the archive to obtain the `Dataset/` directory.

## Embedding Encoder

The provided embeddings are generated using **Qwen3-VL-Embedding-2B**, with **2,048 dimensions** per vector. This applies to the feature vectors in `user_name.pt`, `user_description.pt`, `user_avatar.pt` and `post_text.pt`.

## Feature Alignment and Missing Values

Embedding matrices have shape `[N, 2048]`; rows correspond to the identifiers stored in the same file. Align features by identifier rather than assuming the same row order across files. In `user_name.pt`, `user_description.pt` and `user_avatar.pt`, `mask=false` indicates a missing feature, represented by a zero vector. Use the mask to distinguish missing features from available embeddings. Missing `followers_count` or `following_count` values in `users.pt` are stored as `None`; these indicate missing data, not zero counts. Handle missing values before numeric conversion or normalization.

## Training Group Labels

Each training package includes `labels.pt`, a dictionary with these fields:

- `group_count`: the number of groups eligible for scoring, not the length of the full `groups` list.
- `groups`: group records, including both eligible and excluded records. Each record has `group_id` (identifier within the package), `members` (user identifiers), `eligible_for_scoring` (whether the group contributes to the evaluation target), and `is_other` (whether the record represents OTHER/background users rather than a target group).
- `user_id`: user identifiers for the per-user label representation.
- `group_ids`: lists of group identifiers, aligned row by row with `user_id`. These lists may include excluded groups; filter them using the eligible group identifiers when constructing training targets.

Use only records with `eligible_for_scoring=true` as target groups. OTHER records (`is_other=true`) and groups with fewer than five distinct members are not eligible. All members of an eligible group remain in its member list, even if they also occur in an excluded record. Do not remove such users from the input data or from eligible groups.

The complete `groups` list is retained for reference. For example, `two_events_01/labels.pt` contains 17 records, but `group_count` is 14: one OTHER record and two groups with fewer than five members are excluded. The stored flags define the existing evaluation target; this explanation does not change the labels or scoring rules.

```python
import torch

labels = torch.load("train/two_events_01/labels.pt", map_location="cpu", weights_only=True)
target_groups = [g for g in labels["groups"] if g["eligible_for_scoring"]]
assert all(not g["is_other"] and len(set(g["members"])) >= 5 for g in target_groups)
assert len(target_groups) == labels["group_count"]

# Optional: construct per-user targets using the same eligible group set.
eligible_ids = {g["group_id"] for g in target_groups}
user_targets = {
    uid: [gid for gid in gids if gid in eligible_ids]
    for uid, gids in zip(labels["user_id"], labels["group_ids"])
}
```

A user may belong to multiple target groups or to none. Test packages do not include `labels.pt`. Submitted predictions should contain only the discovered target groups, following the website's prediction schema; do not submit an OTHER group or eligibility flags.

