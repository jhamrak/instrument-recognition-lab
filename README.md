# instrument-recognition-lab

For English readme, [click here](README.en.md).

## A projekt célja

Ez a projekt a programtervező informatikus mesterképzéshez tartozó diplomamunkám része. Célja felépíteni és összehasonlítani két


## Mérföldkövek

1. Hangfájlok beolvasása, megjelenítése.

Egy olyan alkalmas datasetet kerestem, ami ingyen elérhető, megfelelő terjedelmű és a projekt kezdeti fázisában jól használható (több hangszer, hangszerenként több száz fájl, fájlok különböző tulajdonságú monofónikus hangok). Az alábbi linken találhatót választottam:
http://www.philharmonia.co.uk/explore/sound_samples

A letöltött .mp3 fájlokat az Essentia könyvtár EasyLoader-ének segítségével töltöttem be. Az adatok vizualizáláshoz a Matplotlib könyvtárat használtam fel. Hangszerenként egy-egy audió fájl tartalmát rajzoltattam ki. Ezeken jól látszik, hogy előfeldolgozásra szorulnának a megfelelő kezeléshez. A néma részek eltávolítására (silence removal) és amplitúdó normalizációra egyaránt szükség lehet. Erre egy alkalmasabb modell esetében nem biztos, hogy szükségünk lesz.

2. Működő Keras modell építése.

Első körben igyekeztem további feature-ök nélkül, csak a "nyers" adatokkal dolgozni, felépíteni egy modellt. Ekkor a dimenziók száma miatti hibát kaptam. Az ndarray-om shape-je 1, bármilyen modell létrehozásához pedig legalább 2 dimenziójúnak kellene lennie. A probléma valószínűleg a saját adatstruktúrából ndarray-ra való konverzióban rejlett.

Ehelyett áttértem az Essentia könyvtár MFCC függvényéből számított reprezentációjával dolgozni. Ezzel egy működő modellt kaptam végül. Mivel azonban az Essentia könyvtár csak linux alól telepíthető, és a VirtualBoxban futtatott ubuntu környyezet fenntartását macerásnak éreztem, felvetődött az Essentia cseréje. A Librosa nevű könyvtárra esett a választás.

3. Megfelelő polifónikus input felkutatása.

OpenMic.

## Linkek

https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e
