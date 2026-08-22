# Spam Email Detection

A small, production-shaped spam classifier: a scikit-learn pipeline you can train on the public
SMS Spam Collection corpus (or your own labelled CSV), a CLI, and a FastAPI service.

```
raw text -> entity normalisation -> word TF-IDF (1-2 grams)  \
                                                              -> linear classifier -> spam score
                                    char TF-IDF (3-5 grams)  /
```

Why this design: spam is recognisable from its *shape* (links, phone numbers, money amounts,
SHOUTING) as much as its vocabulary, so entities are replaced with placeholder tokens
(`__url__`, `__money__`, `__phone__`, `__allcaps__`, ...) before vectorisation. Character n-grams
on top of word n-grams catch obfuscations such as `V1agra` or `fr€e` that word features miss.

## Results

Held-out test set (20% of the SMS Spam Collection, 1032 messages, `--random-state 42`):

| classifier            | accuracy | precision (spam) | recall (spam) |     F1 | ROC AUC |
| --------------------- | -------: | ---------------: | ------------: | -----: | ------: |
| `linear-svm` (default) |   0.9922 |           0.9762 |        0.9609 | 0.9685 |  0.9990 |
| `logreg`               |   0.9922 |           0.9839 |        0.9531 | 0.9683 |  0.9991 |
| `naive-bayes`          |   0.9893 |           0.9756 |        0.9375 | 0.9562 |  0.9933 |

Reproduce with `spam-detector train --classifier <name>`; every run writes `models/metrics.json`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
```

## Train

```bash
spam-detector train                          # SMS Spam Collection, downloaded and cached
spam-detector train --dataset sample         # tiny bundled dataset, no network needed
spam-detector train --dataset my_emails.csv  # your own CSV with `text` and `label` columns
spam-detector train --target-precision 0.99  # tune the threshold instead of using 0.5
```

Labels may be `spam`/`ham`, `1`/`0` or `true`/`false`. The fitted pipeline is written to
`models/spam_classifier.joblib` together with the decision threshold and the run metadata.

Missing a legitimate email hurts more than letting one spam through, so `--target-precision`
picks the lowest threshold that still reaches the precision you ask for, maximising recall
under that constraint.

## Predict

```bash
spam-detector predict "WINNER!! Claim your free $1000 prize at http://bit.ly/x"
spam-detector predict --file messages.txt --json
cat messages.txt | spam-detector predict
```

```python
from spam_detector import SpamDetector

detector = SpamDetector.load("models/spam_classifier.joblib")
result = detector.predict_one("Your parcel is held, pay the fee at http://track.example")
print(result.label, round(result.spam_probability, 3))
```

## Serve

```bash
spam-detector serve --port 8000          # or: uvicorn spam_detector.api:app
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"messages": ["free entry to win a prize", "see you at lunch"]}'
```

| endpoint       | description                                   |
| -------------- | --------------------------------------------- |
| `GET /health`  | liveness plus whether a model file is present |
| `GET /model`   | metadata and metrics of the loaded model      |
| `POST /predict`| batch classification with spam probabilities  |

Point the service at another artifact with `SPAM_MODEL_PATH=/path/model.joblib`.

## Layout

```
src/spam_detector/
  data.py        dataset download, CSV loading, label normalisation
  preprocess.py  entity placeholders and capitalisation marker
  model.py       TF-IDF feature union + classifier choices
  train.py       train/evaluate/threshold tuning, artifact + metrics writing
  predict.py     SpamDetector: load an artifact and score messages
  api.py         FastAPI app
  cli.py         `spam-detector train|predict|serve`
tests/           pytest suite running on the bundled dataset (no network)
```

## Development

```bash
pytest
ruff check . && ruff format --check .
```
