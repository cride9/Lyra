# Lyra

Egyetlen, nulláról és közösen tanított audio → osu!standard modell. A V1 encoder–decoder felépítésére épül: a mapgenerálás vesztesége az audioencodert is tanítja. Nincs külön audiofázis, befagyasztott encoder vagy kötelező feature-építés.

## Indítás

Nyisd meg a **StartLyra.cmd** fájlt. A kezelőfelület: **http://127.0.0.1:7862**.

Ezen a gépen a meglévő CUDA-s Python-környezetet használhatja. Másik gépen az `Install.ps1` telepíti a függőségeket a felhasználó `AppData/Local/Lyra/venv` mappájába. A telepítő nem másolja be a régi projektet, és nem készít virtuális környezetet a forrásmappában.

- **Tanítás:** adj nevet a futásnak és állítsd be a teljes adatkörök számát. Az Indítás/Folytatás automatikusan a saját `last.pt` mentését használja. A körszám mindig a teljes cél, tehát 3 lezárt kör után 5-re állítva további 2 kör következik.
- **Map készítése:** válassz egy Lyra `best.pt` mentést, zenét, csillagot és stílust. A kimenet közvetlenül importálható `.osz`. Bizonytalan automatikus timinghoz megadható BPM/offset vagy referencia `.osu`.
- **Adatok:** a meglévő teljes dataset manifestjéből minden használható map minden Lyra-ablakát indexeli. Az előkészítés megszakítás után folytatható. A dalok szerinti train/validation/test besorolást változatlanul veszi át.

A tanítás külön háttérfolyamat. A frontend újranyitás, frissítés és a felületi szerver újraindítása után a lemezen tárolt feladathoz és loss-előzményhez kapcsolódik vissza. Átmeneti hálózati hiba esetén automatikusan újrapróbálkozik; a böngészőkapcsolat nem vezérli a tanítás élettartamát. Gép- vagy Python-folyamatleállás után a `last.pt` mentésből lehet folytatni.

## A teljes tanulási napló és statisztikák

Minden lezárt optimizer-update teljes metrikái a futás `metrics.sqlite` adatbázisába kerülnek. A grafikon csak ritkítja a megjelenített pontokat, a tárolt előzményt nem ritkítja. A metrikák: UTC-időbélyeg, lépés, eltelt idő, össz-/token-/beat-loss, validációs értékek, tanulási ráta, gradiensnorma és AMP-skála, update-/betöltési-/tanítási idő, token- és mintasebesség, GPU-memória, cache-találatok/hibák/dekódolások, valamint a teljes dataset bejárásának állapota.

A Tanítás lapon **CSV metrikák** és **Teljes JSONL-előzmény** letöltés van. Az export közvetlenül az adatbázisból készül, így a teljes naplót le lehet tölteni további folyamatos másolatok tárolása nélkül. A szöveges worker-naplók a `jobs` mappában maradnak.

Gépleállás után előfordulhat, hogy a checkpoint régebbi az utolsó naplózott lépésnél. Ilyenkor az újra végzett lépések korábbi metrikái az adatbázis `archived_metrics` táblájában megmaradnak. A normál grafikon és CSV a folytatott modell aktuális tanulási útját mutatja; a teljes JSONL-export a visszaállításkor félretett sorokat is tartalmazza, `rolled_back_update` jelöléssel. Szabályos leállítás/folytatás nem törli a korábbi metrikákat.

## Modell és kontextus

| Beállítás | Érték |
|---|---|
| Együtt tanított paraméterek | **120 612 482** |
| Audioencoder / decoder | 8 / 12 Transformer-réteg |
| Szélesség / fejek / FF | 640 / 10 / 2304 |
| Audio | 22,05 kHz mono, 128 mel sáv, kb. 10 ms melképköz |
| Audioablak | 64 másodperc |
| Friss mapelőzmény | 32 másodperc, legfeljebb 2048 nyers eseménytoken |
| Szokásos generálási szakasz | **24 másodperc** |
| Decoderkeret | 6144 token |
| Távolabbi múlt | Legfeljebb 128 múltbeli mintaleíró; ugyanazon modell tanult projekciója |
| Tanítás | FP16, AdamW, gradient checkpointing; effektív batch 32 |
| Alapprofil | Fizikai batch **4**, 4 CPU-előkészítő, összesen 8 GiB RAM-cache |

A teljes korábbi map megmarad a generálás során. A nyers friss előzmény közvetlenül a decoder tokenfolyamába kerül; a távolabbi múlt összefoglalói csak már elkészült objektumokból készülnek. A tanítás és generálás ugyanazt a kontextusépítőt használja. Az aktív, szakaszhatáron átnyúló sliderek is továbbmennek. Sűrű szakaszban a keret teljes objektumhatáron rövidíthető, és a fennmaradó rész folytatódik.

Ez a korábbi 8 másodperces feladatnál hosszabb közvetlen kapcsolatot biztosít. A tényleges patternkövetkezetességet és játékélményt a tanított modell kimenetein kell majd megítélni; az architektúra és a memóriapróba ezt önmagában nem igazolja.

## A teljes helyi dataset

Az előkészítés **2026. október 19-én elkészült**, további mapkizárás nélkül:

| Részhalmaz | Mapok | Lyra-ablakok |
|---|---:|---:|
| Train | 133 440 | 964 841 |
| Validáció | 7526 | 53 947 |
| Teszt | 6998 | 49 618 |
| Összesen | **147 964** | **1 068 406** |

Ez a meglévő, zenecsoportok szerint kialakított 90/5/5 felosztás. A csoportméretek miatt a tényleges maparányok közelítik a százalékokat. Train körben minden tanítóablak pontosan egyszer kerül sorra, majd új keverés következik; a végső batch kitöltése elkezdheti a következő kört. Nincs csillagsáv-, mapdarabszám- vagy cacheméret miatti train részhalmaz. A validációs és tesztzenék nem kapnak súlyfrissítést.

Három teljes train kör effektív batch 32 mellett **90 454 optimizer-update**. Nincs órakeret, amely ezt félbevágja. A tanulásiráta alapütemezése legalább 100k update-es; resume alatt az eredeti optimizer, skálázó, ütemezés, véletlenállapot és következő ablak megmarad. A validáció a teljes validációs idővonalból választ fix ablakokat, nem kizárólag dalkezdéseket.

**1000 lépésenként külső zenéből készül példamap** a `runs/main/samples/step-<lépés>` mappában, importálható `.osz` formátumban. A main futás példája a `C:/Users/cride/Desktop/music.mp3` **0:50–1:30 közötti, 40 másodperces részlete**. Az eredeti zene pontos PCM-lenyomata nem szerepel a datasetben. A hangrészlet külön, rögzített `.ogg` másolatként az `F:/osu_dataset/_lyra/examples/music-50-90` mappában van; a tanítás `sample_audio_example` beállítása az itteni `example.json` fájlra mutat. Az eredeti asztali fájl későbbi mozgatása nem érinti a példákat. A részlet nem kerül be a training datasetbe.

A modell a timingot és az objektumokat is közvetlenül a zenéből generálja, referenciamap nélkül. A célnehézség 6,5★, stílus Auto, seed 2026, hőmérséklet 0,6; ezek minden példánál azonosak, hogy a súlyok fejlődése összehasonlítható legyen. A mel jellemzők a meglévő, kvótázott SSD-cache-be kerülnek; nincs külön feature- vagy modellfázis. A kivágott hang nulláról indul a játszható csomagban, az eredeti 50–90 másodperces tartomány a `generation.json` fájlban szerepel a lépésszámmal, ellenőrzőösszegekkel, timingjelentéssel és generálási beállításokkal együtt. A korábbi Hakai-példa megmarad, a további periodikus minták a külső részletből készülnek.

A `sample_audio_example` beállítás resume után is megmarad. Külső részlet nélkül a `sample_map_id` választhat datasetmapot, ennek hiányában az első validációs map az alapértelmezés. A példagenerálás hibái részletes `rejected.json` fájlba kerülnek, a tanítás folytatódik. Ha a folyamat a mentés és a hozzá tartozó példa elkészülése között állt le, a hiányzó mintát folytatáskor megpróbálja pótolni. A példák mellett a független validáció külön fut.

## SSD- és RAM-cache

A V2-ben bevált megoldások közül a globális időrácshoz igazított **4096 képkockás FP16 melblokkok**, a feldolgozott mapok, a byte-alapú RAM LRU, a kvótázott SSD-cache, a külön előkészítő folyamatok, a bucketelt batch-ek és az aszinkron GPU-adatmásolás kerültek át. Azonos audióból hiányzó blokkok készítése felvételenként zárolt, ezért a párhuzamos workerek nem dekódolják egyszerre ugyanazt a zenét.

A régi cache használható melblokkjait és timingcéljait **csak olvassa**; az encoder-feature-öket nem használja. A saját SSD-cache alapkerete 180 decimális GB, legalább 8 GiB szabad tárhellyel. Csak saját származtatott bejegyzéseket cserél. Ha egy érték nem menthető cache-be, az adott minta továbbra is használható. A cache soha nem dönti el a dataset tagságát.

Egy valódi validációs ablak ellenőrzött betöltése: hidegen **0,344 s**, RAM-találattal **0,0103 s**, azonos melértékekkel. Ez egyetlen ablak betöltési mérése, nem teljes tanítási gyorsulási ígéret.

## Ellenőrzés az RTX 3050 8 GB-on

Valódi forward/backward/AdamW-frissítések a teljes **64 s / 6144 token** maximális alakon, checkpointinggal; egy bemelegítés után három mért update:

| Fizikai batch | Másodperc / maximális microbatch | Példa / s | PyTorch csúcs |
|---|---:|---:|---:|
| 1 | 1,292 | 0,774 | 2,05 GiB |
| **4** | **4,901** | **0,816** | **3,40 GiB** |
| 8 | 9,911 | 0,807 | 5,37 GiB |

A batch 4 profil gyorsabbnak bizonyult a másik két vizsgált méretnél. A batch 8 után csak 0,55 GiB szabad VRAM maradt, ezért nem ez az alapérték. Ezek maximális szintetikus microbatch-mérések; a tényleges update idejét az effektív batch, az ablakok hossza, a cache és az időszakos validáció/generálás együtt határozza meg.

Automatikus tesztek: causality és KV-cache-egyezés, a mappingveszteség gradientje az audioencoderig, timing-információk elválasztása, train/generálási kontextusegyezés, minden esemény ablaklefedése, dalok szétválasztása, meleg/hideg cache azonossága, pontos optimizeres folytatás, önálló `.osz` export, tartós háttérworker és frontend-visszacsatlakozás.

## Fájlok és parancsok

A projektmappában csak a forrás, a felület, a tesztek, az indító, a telepítő és ez a leírás van. Runtime-adatok:

- `F:/osu_dataset/_lyra/dataset`: tartós Lyra-index és datasetmanifest.
- `F:/osu_dataset/_lyra/runs/main`: saját `last.pt`, `best.pt`, loss-adatbázis és példamapok.
- `F:/osu_dataset/_lyra/jobs`: háttérfeladatok és naplók.
- `F:/osu_dataset/_lyra/generated`: elkészült `.osz` mapok.
- `F:/osu_cache/lyra`: cserélhető mel-/mapcache és ideiglenes exportmunka.

Az eredeti dataset és a régi V1/V2 projekt külön marad. A Lyra sem futtatáshoz, sem generáláshoz nem importálja a régi projekt moduljait. A jelenlegi indító csak annak meglévő Python-környezetét használhatja.

```powershell
python -m lyra ui --port 7862 --open-browser
python -m lyra prepare
python -m lyra benchmark --batch-size 4
python -m lyra train --cycles 3 --batch-size 4 --workers 4
python -m lyra train --cycles 3 --batch-size 4 --resume F:/osu_dataset/_lyra/runs/main/last.pt
python -m lyra generate F:/osu_dataset/_lyra/runs/main/best.pt F:/Music/song.mp3 --stars 5
python -m lyra evaluate F:/osu_dataset/_lyra/runs/main/best.pt
python -m pytest -q -p no:cacheprovider
```

Az új Lyra-architektúra új tanítást igényel. A V1/V2 checkpointok nem nevezhetők át Lyra-checkpointtá. Generáláshoz viszont a saját Lyra-mentés önálló: nem szükséges mellé külön audio-checkpoint, feature-manifest vagy hozzáférés a tanítómapokhoz.
