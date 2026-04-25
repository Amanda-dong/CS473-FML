# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Amanda-dong/CS473-FML/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| src/api/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| src/api/deps.py                     |        3 |        3 |      0% |       3-9 |
| src/api/main.py                     |       16 |        1 |     94% |        19 |
| src/api/routers/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| src/api/routers/datasets.py         |        6 |        0 |    100% |           |
| src/api/routers/health.py           |        5 |        0 |    100% |           |
| src/api/routers/recommendations.py  |      156 |       43 |     72% |46, 210-229, 282, 346, 348, 366-368, 374-376, 387-389, 430, 434-469, 527 |
| src/config/\_\_init\_\_.py          |        3 |        0 |    100% |           |
| src/config/constants.py             |        6 |        0 |    100% |           |
| src/config/settings.py              |       27 |        4 |     85% |23, 27, 31, 35 |
| src/data/\_\_init\_\_.py            |        3 |        0 |    100% |           |
| src/data/audit.py                   |        5 |        0 |    100% |           |
| src/data/base.py                    |       25 |        4 |     84% |27, 38, 41, 44 |
| src/data/etl\_311.py                |       31 |       17 |     45% |25, 46-54, 58-75, 80-81 |
| src/data/etl\_acs.py                |       52 |       34 |     35% |78-82, 87-113, 117, 122-129, 134-143 |
| src/data/etl\_airbnb.py             |       72 |       53 |     26% |83-106, 110, 115-148, 153-160, 167-191 |
| src/data/etl\_boundaries.py         |       51 |       37 |     27% |67, 72, 83-118, 123-156 |
| src/data/etl\_citibike.py           |       64 |       47 |     27% |37, 42-77, 82-89, 94-116 |
| src/data/etl\_google\_trends.py     |        6 |        2 |     67% |    22, 28 |
| src/data/etl\_inspections.py        |       54 |       40 |     26% |33, 51-77, 82-91, 95-144, 149-150 |
| src/data/etl\_licenses.py           |       33 |        9 |     73% |32, 49-61, 74, 89-90 |
| src/data/etl\_permits.py            |       45 |       33 |     27% |25, 36-40, 44-109, 114-115 |
| src/data/etl\_pluto.py              |       43 |       31 |     28% |25, 37-45, 49-96, 101-102 |
| src/data/etl\_runner.py             |       61 |       26 |     57% |50, 52, 58, 88-90, 93-104, 111-121 |
| src/data/etl\_yelp.py               |       45 |       31 |     31% |33, 43-61, 69-84, 89-99 |
| src/data/nta\_layers.py             |       13 |       13 |      0% |      3-29 |
| src/data/quality.py                 |       75 |        9 |     88% |51, 71-77, 98, 123, 137, 159 |
| src/data/registry.py                |       12 |        0 |    100% |           |
| src/features/\_\_init\_\_.py        |        4 |        0 |    100% |           |
| src/features/competition\_score.py  |        7 |        7 |      0% |      3-14 |
| src/features/demand\_signals.py     |       27 |        5 |     81% |44, 51, 53, 84, 87 |
| src/features/feature\_matrix.py     |      205 |       19 |     91% |59, 120, 235, 261, 266, 272, 277, 284, 349, 413, 424-425, 430, 441, 447, 453, 455-459 |
| src/features/ground\_truth.py       |      114 |      114 |      0% |     3-256 |
| src/features/healthy\_gap.py        |        8 |        0 |    100% |           |
| src/features/license\_velocity.py   |       18 |        0 |    100% |           |
| src/features/merchant\_viability.py |        8 |        8 |      0% |      3-17 |
| src/features/microzones.py          |       10 |        0 |    100% |           |
| src/features/rent\_trajectory.py    |       13 |        0 |    100% |           |
| src/features/yelp\_microzones.py    |       31 |       31 |      0% |      3-97 |
| src/features/zone\_crosswalk.py     |       60 |       18 |     70% |105, 134, 138-155, 163 |
| src/models/\_\_init\_\_.py          |        4 |        0 |    100% |           |
| src/models/baselines.py             |        9 |        9 |      0% |      3-20 |
| src/models/cmf\_score.py            |      108 |       48 |     56% |16-17, 23-24, 30-31, 171-180, 193-212, 217, 222-228, 233, 247-253 |
| src/models/explainability.py        |       59 |       12 |     80% |   131-145 |
| src/models/model\_loader.py         |       87 |       29 |     67% |39-61, 68, 76-78, 103-105, 119-121, 139-140 |
| src/models/ranking\_model.py        |       49 |        7 |     86% |15-16, 22-23, 66, 86, 101 |
| src/models/survival\_model.py       |      237 |       63 |     73% |18, 69-73, 75, 91-95, 97-103, 110-115, 118-129, 133, 155-169, 188, 197-215, 224, 247, 264-265, 284, 292-295, 316, 343, 364, 372, 408, 419-420, 460, 472, 485, 531, 557-567 |
| src/models/trajectory\_model.py     |      135 |       64 |     53% |43, 52-53, 97-120, 153-171, 180-211 |
| src/nlp/\_\_init\_\_.py             |        3 |        0 |    100% |           |
| src/nlp/embeddings.py               |      123 |      100 |     19% |32, 48-98, 109-128, 135-147, 157-170, 179-222 |
| src/nlp/gemini\_labels.py           |       95 |       14 |     85% |71, 89-90, 96, 111-112, 145-146, 156, 161, 181, 200, 206-207 |
| src/nlp/neighborhood\_mentions.py   |        4 |        0 |    100% |           |
| src/nlp/review\_aggregates.py       |       43 |       16 |     63% |53, 60, 81-82, 98, 126-152 |
| src/nlp/sentiment.py                |        2 |        2 |      0% |       4-7 |
| src/nlp/subtype\_classifier.py      |       32 |        0 |    100% |           |
| src/nlp/topic\_model.py             |       45 |       45 |      0% |     3-111 |
| src/nlp/white\_space.py             |        3 |        0 |    100% |           |
| src/pipeline/\_\_init\_\_.py        |        3 |        0 |    100% |           |
| src/pipeline/orchestrator.py        |       11 |        1 |     91% |        18 |
| src/pipeline/preflight.py           |      104 |       28 |     73% |48, 63-64, 103-104, 141-142, 208-209, 253-254, 287-288, 300-311, 317-330 |
| src/pipeline/stages.py              |        1 |        0 |    100% |           |
| src/schemas/\_\_init\_\_.py         |        4 |        0 |    100% |           |
| src/schemas/datasets.py             |        5 |        0 |    100% |           |
| src/schemas/requests.py             |        8 |        0 |    100% |           |
| src/schemas/results.py              |       24 |        2 |     92% |     46-66 |
| src/utils/\_\_init\_\_.py           |        4 |        0 |    100% |           |
| src/utils/geospatial.py             |       14 |        7 |     50% |     27-39 |
| src/utils/paths.py                  |        6 |        2 |     67% |     13-14 |
| src/utils/taxonomy.py               |       16 |        1 |     94% |       101 |
| src/validation/\_\_init\_\_.py      |        2 |        0 |    100% |           |
| src/validation/ablation.py          |       75 |       75 |      0% |     3-211 |
| src/validation/backtesting.py       |      149 |       24 |     84% |27-29, 67, 70, 107, 141, 213, 221, 231-243, 249, 259-263, 312, 344-345, 355 |
| **TOTAL**                           | **2881** | **1188** | **59%** |           |


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