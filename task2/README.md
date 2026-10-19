# Task 2: Open Event Detection in Social Media

## Dataset Download

[Download Task 2 Dataset.zip](https://github.com/icdm26SocialChallenge/ICDM2026-Social-Sensemaking/releases/download/datasets-v1.0/Task2-Dataset.zip)

The dataset is distributed through GitHub Releases. Extract the archive to obtain the `Dataset/` directory.

## Embedding Encoder

The provided embeddings are generated using **Qwen3-VL-Embedding-2B**, with **2,048 dimensions** per vector. This applies to the `text` field in `posts.pt`, which contains post-text embeddings rather than raw text.

## Event Type Labels

The `event_type` field uses the following mapping in training labels and submitted predictions:

```json
{
  "0": "NON_EVENT",
  "1": "SOCIAL_AND_LIVELIHOOD_ISSUES",
  "2": "LAW_AND_LITIGATION",
  "3": "INTERNATIONAL_RELATIONS",
  "4": "POLITICAL_AFFAIRS",
  "5": "EMERGENCIES"
}
```

| Label | Meaning |
| --- | --- |
| 0 | Non-event content. |
| 1 | Social and livelihood issues. |
| 2 | Law and litigation. |
| 3 | International relations. |
| 4 | Political affairs. |
| 5 | Emergencies. |

Labels 1–5 represent the five event types. Label 0 identifies non-event posts and is not a sixth event type. The `event_type` value is an integer; the JSON object above uses string keys to show the mapping.
