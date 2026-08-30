 .gitignore                          |   8 [32m+[m[31m-[m
 docs/dataset-loaders.md             |  63 [32m++++++++++++[m
 src/chronis_ml/loaders/__init__.py  |  12 [32m+++[m
 src/chronis_ml/loaders/base.py      |  23 [32m+++++[m
 src/chronis_ml/loaders/example.py   |  21 [32m++++[m
 src/chronis_ml/loaders/globem.py    | 185 [32m++++++++++++++++++++++++++++++++++++[m
 src/chronis_ml/loaders/tiles.py     |  31 [32m++++++[m
 src/chronis_ml/loaders/utils.py     |  87 [32m+++++++++++++++++[m
 src/chronis_ml/schema/__init__.py   |  21 [32m++++[m
 src/chronis_ml/schema/models.py     |  89 [32m+++++++++++++++++[m
 src/chronis_ml/schema/validation.py |  61 [32m++++++++++++[m
 tests/loaders/__init__.py           |   0
 tests/loaders/test_base.py          |  14 [32m+++[m
 tests/loaders/test_globem.py        |  61 [32m++++++++++++[m
 tests/loaders/test_tiles.py         |  24 [32m+++++[m
 tests/loaders/test_utils.py         |  29 [32m++++++[m
 tests/schema/__init__.py            |   0
 tests/schema/test_models.py         |  95 [32m++++++++++++++++++[m
 18 files changed, 821 insertions(+), 3 deletions(-)
