# NBA prognožu robots

Prognozē katras NBA spēles uzvarētāju, uzvaras varbūtību, rezultātu un punktu starpību.
Ņem vērā komandu formu, mājas laukumu, spēles divas dienas pēc kārtas un **savainojumus**.

## Uzstādīšana

```bash
pip install -r requirements.txt
```

## Automātika: vakarā ~21:30, no rīta ~07:00 / ~07:30 / ~08:30

GitHub Actions (`.github/workflows/nba-prognozes.yml`) pats palaiž robotu jūsu repozitorijā.
Nekas nav jādara un nekas nav jāraksta:

- **21:30 (Rīgas laiks): LIKMES.** 3 likmes katrai nakts spēlei.
- **~07:00: REZULTĀTI** (palaišana 06:47 un vēlreiz 07:17). Kuras likmes izgāja, plus kopējā statistika.
- **07:30: WhatsApp ziņa** ar statistikas skaitļiem (nakts, mēnesis, kopā). Iestatīšana aprakstīta zemāk.
- **~08:30: REZULTĀTI vēlreiz** (palaišana 08:17). Noslēdz vēlās rietumkrasta spēles, kas ziemā beidzas pēc 7:00.

GitHub ieplānotos darbus mēdz sākt vēlāk, dažreiz pat līdz 1 stundai, īpaši ap pilnu stundu vai
pusstundu. Tāpēc palaišana ir ieplānota dažas minūtes pirms nominālā laika uz "nepāra" minūtēm, un
rezultāti tiek pārbaudīti vairākas reizes. Vajadzības gadījumā: **Actions → Run workflow**.

### WhatsApp ziņa (CallMeBot, bezmaksas)

1. Atver https://www.callmebot.com/blog/free-api-whatsapp-messages/ un pievieno telefona kontaktiem
   tur norādīto CallMeBot WhatsApp numuru (numurs laiku pa laikam mainās, tāpēc ņem to no šīs lapas).
2. Nosūti tam WhatsApp ziņu: `I allow callmebot to send me messages`
3. Atbildē saņemsi savu **apikey**.
4. GitHub repozitorijā: **Settings → Secrets and variables → Actions → New repository secret**:
   - `WHATSAPP_PHONE` = tavs numurs ar valsts kodu, piem. `+37120000000`
   - `WHATSAPP_APIKEY` = apikey no 3. soļa
5. Pārbaude: **Actions → NBA prognozes → Run workflow → `whatsapp`**.

Pēc katras palaišanas tiek atjaunināta **publiskā lapa**:
**https://edzinslabakais.github.io/nba-prognozes/**
Tā ir piemērota telefonam, tai nav vajadzīga pieteikšanās, un saiti var droši sūtīt draugiem.
Tie paši dati ir arī failā [PROGNOZES.md](PROGNOZES.md), un vēsture glabājas `history/`.

Palaist ārpus grafika: GitHub → **Actions** → **NBA prognozes** → **Run workflow** → izvēlies
`likmes`, `rezultati` vai `lapa` (tikai pārbūvē lapu). GitHub ieplānotie darbi mēdz sākties
5–20 minūtes vēlāk.

Lapai vajadzīgi divi vienreizēji iestatījumi: repozitorijam jābūt **Public**, un
**Settings → Pages → Source** jābūt **GitHub Actions**. Lapa tiek būvēta ar `site.py`
no `web/nba-prognozes.html` dizaina un `export/` datiem.

(`notify.py` var pēc izvēles sūtīt Telegram ziņas, ja tiek iestatīti noslēpumi
`TELEGRAM_BOT_TOKEN` un `TELEGRAM_CHAT_ID`; pašlaik tas netiek izmantots.)

### Noteikumi un statistika

- **Pamatlaiks:** gan NBA, gan NHL likmes vērtē pēc pamatlaika (48 / 60 min). Papildlaiks tiek
  ieskaitīts tikai, ja likmē ir rakstīts "(ieskaitot papildlaiku)". Robots to ņem vērā jau
  varbūtībās: NBA papildlaiks ~4,4 % spēļu, NHL neizšķirts pamatlaikā ~25 %.
- **★ Visticamākā:** katrā spēlē likme, ko bukmeikeri vērtē kā visticamāko. Sezonas simulācijā
  (okt.–nov. 2026, ~1600 likmes) robota paša modeļa "atradumi" pret bukmeikeriem biežāk zaudēja
  (NBA 39 %, NHL 27 %, kad robots vērtēja >10 procentpunktus augstāk), tāpēc ★ izvēlas pēc bukmeikeriem.
- **Mēneša statistika** (no 1. līdz mēneša pēdējai dienai, pēc spēļu datuma): visas likmes,
  ★ visticamākās un Crazy Dazy.

### Arhīvs pa mēnešiem

Īstā statistika sākas 18.11.2026 (novembrī 18.–30.11.), nākamie mēneši ir no 1. līdz pēdējai dienai.
Noslēgtie mēneši ir redzami lapas apakšā sadaļā **Arhīvs pa mēnešiem**.

### NHL

Tas pats hokejam: `python nhl_bot.py likmes` / `python nhl_bot.py rezultati` (automātiski kopā ar NBA).
Lapa: **https://edzinslabakais.github.io/nba-prognozes/nhl/** (lapas augšā pārslēgs NBA | NHL).
Pārskats: [NHL.md](NHL.md), vēsture: `nhl/history/`.

- **Visas NHL likmes vērtē pēc pamatlaika (60 min).** Tikai likmes ar norādi
  "(ieskaitot papildlaiku)" ietver papildlaiku un bullīšus (bullīšu uzvarētājam +1 vārti).
- Modelis: komandu uzbrukuma/aizsardzības reitingi no pamatlaika vārtiem → Puasona sadalījums
  ar neizšķirtu korekciju. Ņem vērā startējošos vārtsargus (ESPN "probable/confirmed"),
  savainojumus, mājas laukumu un spēli otrajā dienā pēc kārtas, un tiek sajaukts ar bukmeikeru
  koeficientiem (80 %).
- Kalibrēts uz 2025-26 sezonu (`python nhl_backtest.py`, parametri `nhl_params.json`).
  Neizšķirts pamatlaikā: modelis 25,1 %, realitāte 25,0 %. Šajā sezonā robots + bukmeikeri
  uzvarētāju uzmin 61,5 % gadījumu, bukmeikeri vieni paši 60,3 %.

### Drošās likmes

Katru vakaru vismaz 3 likmes (līdz 5, ja tās ir ≥ 56 % drošas), katra no citas spēles, ar aptuveno
koeficientu **≥ 1.7**. Drošumu nosaka **bukmeikeru vērtējums**: simulācijā šīs likmes izgāja 51–53 %
gadījumu, bet pēc robota modeļa izvēlētās tikai 42–47 %. Robota vērtējums ir parādīts blakus
salīdzināšanai. Ja vairākas līnijas ir gandrīz vienlīdz drošas (līdz 2 procentpunktiem), tiek ņemta
tā, kurai ir augstākais koeficients.

### Crazy Dazy

Katru vakaru robots no visām nakts spēlēm izvēlas arī 3 likmes ar koeficientu **4.0 vai vairāk**
(katru no citas spēles), kurām ir vislielākā iespēja iziet. Tās var būt lieli handikapi,
virs/zem līnijas, komandas punkti, uzvaras starpības diapazoni, kombinācijas (uzvarētājs + virs/zem)
vai lieli underdogi. Tās tiek noslēgtas no rīta, un tām ir sava statistika (lapas apakšā).
Pie koeficienta 4 reāla iespēja ir ap 20–23 %, tātad parasti iziet 1 no 4–5.

## Prognožu spēlei: LIKMES un REZULTĀTI

```bash
python bot.py likmes        # 3 likmes katrai šīs nakts spēlei, koef. >= 2.1, drošība 1-100
python bot.py rezultati     # nākamajā dienā: kuras likmes izgāja + kopējā statistika
# (to pašu automātiski dara GitHub Actions, sk. augstāk)
python bot.py likmes --min-odds 2.5
```

"Drošība" ir robota aprēķinātā varbūtība, ka likme izies. Pie koeficienta 2.1 tā praktiski
nevar būt lielāka par ~44 %: bukmeikers 2.1 maksā tikai tad, ja pats vērtē varbūtību zem ~45 %.
Robots katrai spēlei meklē **drošāko** līniju no 3 kategorijām (handikaps/uzvarētājs,
kopējie punkti, komandas punkti), kas vēl dod koeficientu ≥ 2.1. Optibet koeficientu
robots nezina, to novērtē kā 0,93 / varbūtība. "Godīgais koef." parāda, cik likme ir patiesībā vērta.
Ja Optibet dod vairāk, tā ir laba likme.

Detalizētāk: `python bets.py` parāda visu likmju cenu lapu un izveido `odds/DATUMS.csv`.
Ieraksti tur īstos Optibet koeficientus un palaid `python bets.py --odds odds/DATUMS.csv`.
Robots tad sakārto likmes pēc sagaidāmā ieguvuma.

## Lietošana (prognozes)

```bash
python predict.py                      # šodienas spēles (ASV datums)
python predict.py --detail             # + katras spēles faktori (reitingi, savainojumi)
python predict.py --date 2026-11-20    # konkrēta diena
python predict.py --days 3             # 3 dienas uz priekšu
python predict.py --refresh            # svaigākais savainojumu saraksts (ignorē kešu)
python predict.py --market-weight 0.9  # sajaukt ar bukmeikeru līniju (sk. zemāk)
python backtest.py                     # pārbaudīt precizitāti vēsturē un pārkalibrēt (~10 min)
```

Rezultāti tiek saglabāti arī `predictions/YYYY-MM-DD.csv`.
Laiki norādīti Rīgas laikā; "datums" ir ASV datums (NBA kalendārs).

## Kā tas strādā

Dati: ESPN publiskais API (rezultāti, box score, savainojumu saraksts, sastāvi, likmju līnijas).

1. **Komandu reitingi**: uzbrukuma un aizsardzības reitings katrai komandai, aprēķināts ar
   svērto regresiju no visām sezonas spēlēm, ņemot vērā pretinieka stiprumu. Nesenākas
   spēles sver vairāk (svars samazinās uz pusi ik pēc 40 dienām). Sezonas sākumā reitingi tiek
   pievilkti pagājušās sezonas virzienā.
2. **Mājas laukums** (~+2 p.) un **otrā spēle pēc kārtas** (~−2 p.) tiek aprēķināti no datiem.
3. **Savainojumi / sastāvs**: katram spēlētājam aprēķina vērtību punktos no viņa statistikas
   (Game Score uz 36 min virs aizvietotāja līmeņa). Komandas reitings jau atspoguļo, kas
   spēlēja pēdējās 10 spēlēs, tāpēc korekcija ir
   `vērtība × (P(spēlēs šodien) − daļa no pēdējām 10 spēlēm, ko spēlēja)`.
   Ja zvaigzne ir "Out", komanda zaudē punktus. Ja zvaigzne atgriežas pēc ilgas prombūtnes,
   komanda tos iegūst. Statusi: Out = 0 %, Doubtful = 20 %, Questionable = 50 %,
   Day-To-Day = 60 %.
4. Starpība tiek pārvērsta varbūtībā ar normālo sadalījumu (σ = 12,5 p.).
5. Likmēm robots ņem 90 % bukmeikeru līnijas un 10 % sava modeļa (tas vēsturē bija precīzākais),
   bet kopējiem punktiem izmanto bukmeikeru kopsummu. Rezultātu izkliede ap prognozi:
   13,75 p. starpībai un 18,25 p. kopsummai. Tā ir pārbaudīta uz visām handikapa un virs/zem
   līnijām 2025-26 sezonā (piemēram, prognozēts 37,2 %, faktiski 36,8 %).

Parametri ir automātiski kalibrēti ar `backtest.py` un glabājas `params.json`.

## Cik precīzs tas ir (backtest, katra diena prognozēta tikai ar iepriekšējiem datiem)

| | Uzminēti uzvarētāji | Vid. kļūda starpībā |
|---|---|---|
| 2025-26, tikai reitingi | 68,6 % | 11,43 p. |
| 2025-26, + sastāvi/savainojumi | 68,6 % | 11,23 p. |
| 2025-26, bukmeikeru noslēguma līnija | 68,3 % | 11,20 p. |
| 2026-27 līdz 18.11. (210 spēles), robots | 70,0 % | 12,12 p. |
| 2026-27 līdz 18.11., bukmeikeri | 72,4 % | 11,26 p. |

Godīgs secinājums: robots uzminēto uzvarētāju skaita ziņā ir tuvu bukmeikeriem, bet
bukmeikeru līnija joprojām ir nedaudz precīzāka. Vēsturē labākais rezultāts bija ar
`--market-weight 0.9` (90 % tirgus, 10 % robots). Robots var būt noderīgs, lai redzētu,
**kāpēc** viena komanda ir favorīte, un lai atrastu spēles, kur robots un bukmeikeri
būtiski nepiekrīt.

Piezīme: vēsturiskajā testā savainojumiem izmantots faktiskais sastāvs (kas reāli spēlēja),
tāpēc ar reālu savainojumu sarakstu pirms spēles rezultāts būs nedaudz vājāks.

## Ierobežojumi

- Savainojumu saraksts mainās līdz pat spēles sākumam, tāpēc palaid ar `--refresh` neilgi pirms spēlēm.
- Spēlētāju vērtība balstās uz box score (Game Score), nevis uz +/- metrikām (RAPM, EPM).
  Tāpēc aizsardzības speciālisti var būt novērtēti par zemu.
- Netiek ņemti vērā ceļojumi, laika joslas, augstums virs jūras līmeņa (Denver) un motivācija
  (piemēram, komandas sezonas beigās "tankojot").
