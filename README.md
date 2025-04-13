# О решении

За основу нашего решения для генерации молекул мы взяли [SPMM](https://www.nature.com/articles/s41467-024-46440-3):

SPMM представляет собой мультимодальную модель, объединяющую структурные и свойственные данные молекул в едином пространстве представлений:
- SPMM позволяет генерировать структуры молекул (SMILES) на основе заданных свойств, что является ключевым требованием задачи
- Возможность учитывать множество свойств одновременно без необходимости повторного обучения модели.
- Модель обучается на данных, где свойства и структура молекул представлены как отдельные модальности. Это позволяет выявлять сложные зависимости между структурой молекулы и ее антиокислительными характеристиками
- SPMM демонстрирует высокую точность при генерации молекул с заданными свойствами и предсказании свойств для новых структур. Это делает ее подходящей для поиска новых антиоксидантов с целевыми характеристиками

Для ретросинтетического анализа использовали `rxn4chemestry`. Для проверки на стабильность - `MMFF94`. Для оптимизации PDSC использовали генетический алгоритм (в качестве суррогатной модели дообучили `SPMM-регрессор` на данных, собранных с лидерборда)

# Результаты

- [код инференса spmm](https://github.com/l1ghtsource/spmm-neftecode/blob/main/spmm-inference.ipynb)
- [модификация кода для генерации smiles по pv](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/d_pv2smiles_single.py)
- [генетический алгоритм для максимизации oit](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/spmm_ga_optimization.py)
- [pdsc предиктор](https://github.com/l1ghtsource/spmm-neftecode/blob/main/SPMM/spmm_finetune_pdsc.py)
- [pdsc предиктор (вторая версия с ridge)](https://github.com/l1ghtsource/spmm-neftecode/blob/main/regressor.ipynb)
- [код ретросинтетического анализа](https://github.com/l1ghtsource/spmm-neftecode/blob/main/retrosynthesis.ipynb)
- [веб-сервис для работы с ретросинтетическим анализом](https://github.com/l1ghtsource/spmm-neftecode/blob/main/app.py)
- [код проверки молекулы на стабильность](https://github.com/l1ghtsource/spmm-neftecode/blob/main/stability.py)

Сгенерированные молекулы: [тык](https://github.com/l1ghtsource/spmm-neftecode/blob/main/generated_moleculas.csv)

Презентация: [тык](https://disk.yandex.ru/d/zmH7JqOOpjsojQ)

Метрики регрессора PDSC:

```python
Ridge Metrics
- MSE: 852.9046
- RMSE: 29.2045
- MAE: 18.8468
- R²: 0.8655
```
