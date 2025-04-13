За основу нашего решения для генерации молекул мы взяли [SPMM](https://www.nature.com/articles/s41467-024-46440-3):

- [код инференса spmm](https://github.com/l1ghtsource/spmm-neftecode/blob/main/spmm-inference.ipynb)
- [модификация кода для генерации smiles по pv](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/d_pv2smiles_single.py)
- [генетический алгоритм для максимизации oit](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/spmm_ga_optimization.py)
- [pdsc предиктор](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/spmm_finetune_pdsc.py)
- [pdsc предиктор (вторая версия с ridge)](https://github.com/l1ghtsource/spmm-neftecode/blob/main/regressor.ipynb)
- [код ретросинтетического анализа](https://github.com/l1ghtsource/spmm-neftecode/blob/main/retrosynthesis.ipynb)
- [веб-сервис для работы с ретросинтетическим анализом](https://github.com/l1ghtsource/spmm-neftecode/blob/main/app.py)
