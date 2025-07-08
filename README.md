# Viacvrstvová neurónová sieť – predikcia hodnôt neznámej funkcie

Projekt spočíval v naprogramovaní viacvrstvovej neurónovej siete a jej následného tréningu.  
Sieť mala predpovedať hodnoty neznámej funkcie, ktorých výstupy boli v rozsahu [-1, 3].

## Najdôležitejšie ladené parametre

- počet skrytých vrstiev  
- rozmery jednotlivých vrstiev  
- learning rate  
- aktivačné funkcie vrstiev  
- stratová funkcia  
- momentum

## Obsah súborov

- **`projekt.py`** – hlavný súbor projektu, využíva `utils.py` na vykresľovanie výsledkov a vývoja výšky straty

- **`mlp_train.txt`** – obsahuje 1771 trénovacích dát, ktoré boli rozdelené na trénovaciu a testovaciu vzorku

- **`model_weights.txt`** – uložený predtrénovaný model vrátane parametrov, rozmerov a samotných váh

- **`best.txt`** – najlepší natrénovaný model (15 minút tréningu, cca 2000 epoch)

- **`readme_report_lucina.pdf`** – popisuje proces a odôvodnenie voľby hodnôt parametrov

## Výsledky

Model `best.txt` dosahuje priemerné chyby:

- Final MSE on test data: `0.0079`  
- Final MSE on all data: `0.0073`

## Porovnanie hodnôt trénovacích dát

355 dátových bodov = 20 % z 1771

V hornej časti sú skutočné hodnoty, v dolnej predikované:

![image](https://github.com/user-attachments/assets/6afa9ad6-d144-40a1-ac49-874d9defa02d)
