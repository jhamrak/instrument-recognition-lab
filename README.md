# instrument-recognition-lab

For English readme, [click here](README.en.md).

## A projekt célja

Ez a projekt a programtervező informatikus mesterképzéshez tartozó diplomamunkám része. A projekt keretében automatikus zenei hangszerfelismeréssel kapcsolatos mély tanulási és hagyományos gépi tanulási módszerek feltárását és összehasonlítását valósítottam meg.
- [Diplomamunka](docs/thesis.pdf)
- [Témabejelentő](docs/topicdesc.pdf)

## Beüzemelés

### Adatok előkészítése

Miután clone-oztuk a repot, töltsünk le egy datasetet.
- Teljes OpenMIC dataset letöltése: https://zenodo.org/record/1432913 VAGY
- Redukált OpenMIC dataset letöltése: TODO csak a kiválogatott hangszerekre redukált dataset
A letöltött fájlt csomagoljuk ki a repo gyökerébe.

Ha bemenetként melspectogram, vagy MFCC reprezentációt szeretnénk használni, akkor a dataset kicsomagolása után futtassuk a megfelelő adattranszformáló scriptet.
- OpenMIC melspectogramra való tarnszformáló script: [melspec.py](melspec.py).
- OpenMIC MFCC-re való transzformáló script: [mfcc.py](mfcc.py).

### Futtatás

A tanító algoritmus futtatást a [run.py](run.py) scripttel tehetjük meg. Ennek a következő opcionális paraméterei vannak:
- --mode (default DL): ha értéke DL, akkor Deep Learning, ha értéke ML, akkor Machine Learning algoritmussal dolgozunk
- --data (default VGG): az adathalmaz (VGG / MEL / MFCC)
- --threshold (default 0.5): A hangszer jelenlétét jelző küszöbszám
- --lr (default 0.0001): a Deep Learning módszerhez tartozó learning rate paraméter
- --epochs (default 10): a Deep Learning módszerhez tartozó tanítási iterációk száma

## Kiértékelés

Futás közben a konzolon megjelenő információkon kívül fájlba is írunk. Minden futtatáskor létrejön egy mappa a /logs mappán belül a futtatás időpontjával és fő paramétereinek nevével.
Itt található:
- Hangszerenként egy ábra a vonatkozó tanulási görbével
- Egy szöveges "log.txt" fájl, melynek tartalma:
  - a futtatás bemeneti paraméterei
  - a Deep Learning modellünk paraméterei
  - hangszerenkénti teljes táblázatos kiértékelés
