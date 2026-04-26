# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| src/api/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| src/api/deps.py                     |        3 |        3 |      0% |       3-9 |
| src/api/main.py                     |       16 |        1 |     94% |        19 |
| src/api/routers/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| src/api/routers/datasets.py         |        7 |        0 |    100% |           |
| src/api/routers/health.py           |        5 |        0 |    100% |           |
| src/api/routers/recommendations.py  |      156 |       43 |     72% |46, 210-229, 282, 346, 348, 366-368, 374-376, 387-389, 430, 434-469, 527 |
| src/config/\_\_init\_\_.py          |        3 |        0 |    100% |           |
| src/config/constants.py             |        6 |        0 |    100% |           |
| src/config/settings.py              |       27 |        0 |    100% |           |
| src/data/\_\_init\_\_.py            |        3 |        0 |    100% |           |
| src/data/audit.py                   |        5 |        0 |    100% |           |
| src/data/base.py                    |       25 |        0 |    100% |           |
| src/data/etl\_311.py                |       31 |        0 |    100% |           |
| src/data/etl\_acs.py                |       52 |        7 |     87% |81-82, 128-129, 136-138 |
| src/data/etl\_airbnb.py             |       72 |       12 |     83% |155-159, 177-178, 185-191 |
| src/data/etl\_citibike.py           |       64 |       11 |     83% |95-100, 107-111 |
| src/data/etl\_google\_trends.py     |        6 |        0 |    100% |           |
| src/data/etl\_inspections.py        |       54 |        5 |     91% |73-75, 149-150 |
| src/data/etl\_licenses.py           |       33 |        3 |     91% | 74, 89-90 |
| src/data/etl\_permits.py            |       45 |        2 |     96% |   114-115 |
| src/data/etl\_pluto.py              |       43 |        4 |     91% |56-57, 101-102 |
| src/data/etl\_runner.py             |       61 |       11 |     82% |88-90, 97-104, 112 |
| src/data/etl\_yelp.py               |      123 |       56 |     54% |72, 77, 99-101, 110, 145, 151, 167-188, 192-252 |
| src/data/quality.py                 |       75 |        3 |     96% |71, 77, 123 |
| src/data/registry.py                |       12 |        0 |    100% |           |
| src/features/\_\_init\_\_.py        |        4 |        0 |    100% |           |
| src/features/competition\_score.py  |        7 |        0 |    100% |           |
| src/features/demand\_signals.py     |       27 |        0 |    100% |           |
| src/features/feature\_matrix.py     |      205 |       18 |     91% |59, 120, 235, 266, 272, 277, 284, 349, 413, 424-425, 430, 441, 447, 453, 455-459 |
| src/features/ground\_truth.py       |      114 |        0 |    100% |           |
| src/features/healthy\_gap.py        |        8 |        0 |    100% |           |
| src/features/license\_velocity.py   |       18 |        0 |    100% |           |
| src/features/merchant\_viability.py |        8 |        0 |    100% |           |
| src/features/microzones.py          |       10 |        0 |    100% |           |
| src/features/rent\_trajectory.py    |       13 |        0 |    100% |           |
| src/features/zone\_crosswalk.py     |       60 |        4 |     93% |146, 149-152 |
| src/models/\_\_init\_\_.py          |        4 |        0 |    100% |           |
| src/models/baselines.py             |        9 |        0 |    100% |           |
| src/models/cmf\_score.py            |      108 |       11 |     90% |16-17, 23-24, 30-31, 177-178, 226-228 |
| src/models/explainability.py        |       59 |        0 |    100% |           |
| src/models/model\_loader.py         |       87 |        9 |     90% |60-61, 103-105, 119-120, 139-140 |
| src/models/ranking\_model.py        |       49 |        5 |     90% |15-16, 22-23, 66 |
| src/models/survival\_model.py       |      247 |       45 |     82% |16-18, 25, 76-80, 86, 93-94, 109-115, 122-127, 130-141, 167-181, 210-221, 276-277, 296, 304-307, 328, 376, 384, 543 |
| src/models/trajectory\_model.py     |      135 |        7 |     95% |116, 158-160, 188-189, 208 |
| src/nlp/\_\_init\_\_.py             |        3 |        0 |    100% |           |
| src/nlp/embeddings.py               |      123 |       20 |     84% |70-98, 122, 166 |
| src/nlp/gemini\_labels.py           |      157 |       32 |     80% |142-143, 165-166, 173-174, 177, 185-206, 210-211, 226-229, 232-233, 250-252, 293, 298, 316, 328, 334-335 |
| src/nlp/neighborhood\_mentions.py   |        4 |        0 |    100% |           |
| src/nlp/review\_aggregates.py       |       77 |       32 |     58% |   172-292 |
| src/nlp/sentiment.py                |        2 |        0 |    100% |           |
| src/nlp/subtype\_classifier.py      |       32 |        0 |    100% |           |
| src/nlp/topic\_model.py             |       45 |        3 |     93% | 40, 57-58 |
| src/nlp/white\_space.py             |        3 |        0 |    100% |           |
| src/pipeline/\_\_init\_\_.py        |        3 |        0 |    100% |           |
| src/pipeline/orchestrator.py        |       11 |        1 |     91% |        18 |
| src/pipeline/preflight.py           |      104 |        6 |     94% |208-209, 253-254, 287-288 |
| src/pipeline/stages.py              |        1 |        0 |    100% |           |
| src/schemas/\_\_init\_\_.py         |        4 |        0 |    100% |           |
| src/schemas/datasets.py             |        6 |        0 |    100% |           |
| src/schemas/requests.py             |        9 |        0 |    100% |           |
| src/schemas/results.py              |       25 |        0 |    100% |           |
| src/utils/\_\_init\_\_.py           |        4 |        0 |    100% |           |
| src/utils/geospatial.py             |       14 |        0 |    100% |           |
| src/utils/paths.py                  |        6 |        0 |    100% |           |
| src/utils/taxonomy.py               |       16 |        1 |     94% |       101 |
| src/validation/\_\_init\_\_.py      |        2 |        0 |    100% |           |
| src/validation/ablation.py          |       75 |        0 |    100% |           |
| src/validation/backtesting.py       |      149 |        5 |     97% |249, 312, 344-345, 355 |
| **TOTAL**                           | **2974** |  **360** | **88%** |           |


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