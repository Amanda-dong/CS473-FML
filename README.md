# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| src/api/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| src/api/deps.py                     |        3 |        0 |    100% |           |
| src/api/main.py                     |       16 |        0 |    100% |           |
| src/api/routers/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| src/api/routers/datasets.py         |        7 |        0 |    100% |           |
| src/api/routers/health.py           |        5 |        0 |    100% |           |
| src/api/routers/recommendations.py  |      246 |       19 |     92% |40-41, 61, 66-68, 73, 79-80, 89, 251-253, 269-273, 286 |
| src/config/\_\_init\_\_.py          |        3 |        0 |    100% |           |
| src/config/constants.py             |        6 |        0 |    100% |           |
| src/config/settings.py              |       27 |        0 |    100% |           |
| src/data/\_\_init\_\_.py            |        3 |        0 |    100% |           |
| src/data/audit.py                   |        5 |        0 |    100% |           |
| src/data/base.py                    |       25 |        0 |    100% |           |
| src/data/etl\_311.py                |       31 |        0 |    100% |           |
| src/data/etl\_acs.py                |       43 |        0 |    100% |           |
| src/data/etl\_airbnb.py             |       53 |        1 |     98% |       114 |
| src/data/etl\_citibike.py           |       64 |        0 |    100% |           |
| src/data/etl\_inspections.py        |       54 |        0 |    100% |           |
| src/data/etl\_licenses.py           |       33 |        0 |    100% |           |
| src/data/etl\_permits.py            |       45 |        0 |    100% |           |
| src/data/etl\_pluto.py              |       43 |        0 |    100% |           |
| src/data/etl\_runner.py             |       61 |        0 |    100% |           |
| src/data/etl\_yelp.py               |      123 |        0 |    100% |           |
| src/data/quality.py                 |       74 |        1 |     99% |        71 |
| src/data/registry.py                |       11 |        0 |    100% |           |
| src/features/\_\_init\_\_.py        |        4 |        0 |    100% |           |
| src/features/competition\_score.py  |        7 |        0 |    100% |           |
| src/features/demand\_signals.py     |       27 |        0 |    100% |           |
| src/features/feature\_matrix.py     |      209 |        5 |     98% |297, 305, 463, 469, 475 |
| src/features/ground\_truth.py       |      114 |        0 |    100% |           |
| src/features/healthy\_gap.py        |       10 |        0 |    100% |           |
| src/features/license\_velocity.py   |       18 |        0 |    100% |           |
| src/features/merchant\_viability.py |        8 |        0 |    100% |           |
| src/features/microzones.py          |       10 |        0 |    100% |           |
| src/features/rent\_trajectory.py    |       13 |        0 |    100% |           |
| src/features/zone\_crosswalk.py     |       60 |        0 |    100% |           |
| src/models/\_\_init\_\_.py          |        4 |        0 |    100% |           |
| src/models/baselines.py             |        9 |        0 |    100% |           |
| src/models/cmf\_score.py            |      108 |        6 |     94% |16-17, 23-24, 30-31 |
| src/models/explainability.py        |       59 |        0 |    100% |           |
| src/models/model\_loader.py         |       87 |        0 |    100% |           |
| src/models/ranking\_model.py        |       49 |        4 |     92% |15-16, 22-23 |
| src/models/survival\_model.py       |      247 |        5 |     98% |16-18, 25, 543 |
| src/models/trajectory\_model.py     |      135 |        0 |    100% |           |
| src/nlp/\_\_init\_\_.py             |        3 |        0 |    100% |           |
| src/nlp/embeddings.py               |      123 |        0 |    100% |           |
| src/nlp/gemini\_labels.py           |      157 |        0 |    100% |           |
| src/nlp/neighborhood\_mentions.py   |        4 |        0 |    100% |           |
| src/nlp/review\_aggregates.py       |       77 |        0 |    100% |           |
| src/nlp/sentiment.py                |        2 |        0 |    100% |           |
| src/nlp/subtype\_classifier.py      |       32 |        0 |    100% |           |
| src/nlp/topic\_model.py             |       45 |        0 |    100% |           |
| src/nlp/white\_space.py             |        3 |        0 |    100% |           |
| src/pipeline/\_\_init\_\_.py        |        3 |        0 |    100% |           |
| src/pipeline/orchestrator.py        |       11 |        0 |    100% |           |
| src/pipeline/preflight.py           |      104 |        0 |    100% |           |
| src/pipeline/stages.py              |        1 |        0 |    100% |           |
| src/schemas/\_\_init\_\_.py         |        4 |        0 |    100% |           |
| src/schemas/datasets.py             |        6 |        0 |    100% |           |
| src/schemas/requests.py             |        9 |        0 |    100% |           |
| src/schemas/results.py              |       25 |        0 |    100% |           |
| src/utils/\_\_init\_\_.py           |        4 |        0 |    100% |           |
| src/utils/geospatial.py             |       14 |        0 |    100% |           |
| src/utils/paths.py                  |        6 |        0 |    100% |           |
| src/utils/taxonomy.py               |       16 |        0 |    100% |           |
| src/validation/\_\_init\_\_.py      |        2 |        0 |    100% |           |
| src/validation/ablation.py          |       75 |        0 |    100% |           |
| src/validation/backtesting.py       |      149 |        0 |    100% |           |
| **TOTAL**                           | **3034** |   **41** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Amanda-dong/CS473-FML/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Amanda-dong/CS473-FML/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FAmanda-dong%2FCS473-FML%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.