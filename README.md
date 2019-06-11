# instrument-recognition-lab

## Project vision

A project célja egy olyan deep learning modell betanítása, amely képes egy nyers audio fájlból (pl. .wav) megállapítani a benne megszólaló hangszer(eke)t. Első körben csak egy hangszert tartalmazó hangfájlokra, és amennyiben itt megfelelő eredményeket produkál a modell, akkor polifónikus audióra is ki lehetne terjeszteni.


## Milestones

1. Read audio files, plot some data.

Egy olyan alkalmas datasetet kerestem, ami ingyen elérhet, megfelelő terjedelmű és a projekt számára jól használható (több hangszer, hangszerenként több száz fájl, fájlok különböző tulajdonságú monofónikus hangok). Az alábbi linken találhatót választottam:
http://www.philharmonia.co.uk/explore/sound_samples

A letöltött .mp3 fájlokat az Essentia könyvtár EasyLoader-ének segítségével töltöttem be. Az adatok vizualizáláshoz a Matplotlib könyvtárat használtam fel. Hangszerenként egy-egy audió fájl tartalmát rajzoltattam ki. Ezeken jól látszik, hogy előfeldolgozásra szorulnának a megfelelő kezeléshez. A néma részek eltávolítására (silence removal) és amplitúdó normalizációra egyaránt szükség lehet. Ezt az első működő modell megalkotása utánra tervezem megtenni.

2. Build some model with keras.

Első körben igyekeztem további feature-ök nélkül, csak a "nyers" adatokkal dolgozni, felépíteni egy modellt. Jelenleg a dimenziók száma miatti hibát kapok. Az ndarray-om shape-je 1, bármilyen modell létrehozásához pedig legalább 2 dimenziójúnak kellene lennie. A probléma valószínűleg a saját adatstruktúrából ndarray-ra való konverzióban rejlik.

3. Collect better features.

MFCC illetve FFT algoritmusok benne foglaltatnak az Essentia kkkönyvtárban, ezek eredményeit valószínűleg hozzá lehetne csatolni az eredeti datasethez.

4. Poliphonic inputs
