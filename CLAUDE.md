# NBA likmju robots — instrukcijas Claude

Lietotājs raksta latviski. Atbildi latviski.

## Komandas čatā

- **LIKMES** (jebkurā reģistrā) → palaid `python3 bot.py likmes --refresh` šajā mapē un parādi
  rezultātu: katrai šīs nakts spēlei 3 likmes ar koeficientu ≥ 2.1 un "drošību" 1–100.
  Ja lietotājs norāda citu min. koeficientu, pievieno `--min-odds X`.
  Pēc tam īsi (2–4 teikumos) izcel interesantāko: spēles ar lieliem savainojumiem vai kur
  robots būtiski nepiekrīt bukmeikeriem. Neizdomā varbūtības — izmanto tikai skripta skaitļus.
- NHL: tās pašas komandas ar `python3 nhl_bot.py likmes --refresh` / `python3 nhl_bot.py rezultati`
  (vēsture `nhl/history/`, pārskats `NHL.md`). NHL likmes vērtē pēc pamatlaika, izņemot
  likmes ar norādi "(ieskaitot papildlaiku)".
- **REZULTĀTI** → palaid `python3 bot.py rezultati` un parādi kopsavilkumu (kuras likmes
  izgāja, dienas un kopējā statistika).
- Likmju vēsture glabājas `history/*.json`. Pēc LIKMES un REZULTĀTI saglabā izmaiņas
  (git commit + push), citādi vēsture pazudīs līdz ar sesiju.

## Automātika

LIKMES (~21:30, cron 21:17), REZULTĀTI (~07:00: 06:47 + 07:17, un ~08:30: 08:17) un WhatsApp statistiku
(~07:30: 07:33, `whatsapp.py`, CallMeBot; Rīgas laiks) automātiski palaiž GitHub Actions. GitHub cron var kavēties
līdz ~1 h (īpaši :00/:30), tāpēc laiki ir nepāra minūtēs un rezultāti tiek palaisti vairākkārt
(`.github/workflows/nba-prognozes.yml` + `scheduler.py`). Papildus ik pēc 5 min (`*/5`, atsevišķa rinda `nba-live`)
`scheduler.live_bots()` pārbauda, vai kādai likmei spēlei jau vajadzētu būt beigušai, un palaiž `rezultati_live` tikai
tām līgām; lapu publicē tikai, ja kaut kas mainījies. Saglabāšana sāk no jaunākā `origin/main` un pārrēķina, ja push neizdodas. Pēc tam tiek publicēta publiskā lapa
https://edzinslabakais.github.io/nba-prognozes/ un …/nhl/ (`site.py` → `site/`, GitHub Pages), kā arī atjaunināts
`PROGNOZES.md`. Claude tajā nav vajadzīgs. Galvenā lapa ir šī publiskā lapa. Pirms rokas palaišanas veic `git pull`, jo
Actions pievieno savus commit.

## Telefona lapa "NBA prognozes" (claude.ai, tikai rokas režīmā)

URL: https://claude.ai/artifact/CrSWZsA6HApj7T1AwenP4x (lapas avots: `web/nba-prognozes.html`).
Lapa rāda datus no artifact datubāzes. GitHub Actions to NEatjaunina. Kad Claude čatā palaiž
LIKMES vai REZULTĀTI, atjaunini to
(skripts pats izveido `export/`):
1. ArtifactData `batch` ar `set`: katram `export/nights/<datums>.json` → collection `nights`,
   doc_id `<datums>`; `export/stats.json` → collection `meta`, doc_id `stats`.
   Esošiem dokumentiem vajag `if_version`, tāpēc vispirms nolasi versijas ar `list` (`nights`, `meta`).
2. Pārbaudi, ka rakstīšana izdevās. Lapu nepublicē no jauna, ja nemainās tās dizains.

## Lapas valodas (LV / EN)

Lapai ir latviešu un angļu versija (`web/nba-prognozes.html`: vārdnīca `I18N`, `NOTES`, `CAT_EN`,
tulkotāji `trBet/trScore/trNote/trLine`). **Lietotāja noteikums:** jaunas lapas izmaiņas veic tikai
latviski. Angļu tekstus (`I18N.en`, `NOTES.en`, `CAT_EN`, tulkotājus) atjaunini tikai tad, ja
lietotājs to skaidri prasa. Ja jaunam tekstam nav EN tulkojuma, `t()` rāda latvisko.

## Struktūra

- `espn.py` — ESPN API + kešs (`cache/`), `data.py` — tabulas, `model.py` — reitingi un savainojumi
- `predict.py` — spēļu prognozes, `bets.py` — likmju varbūtības (cenu lapa / Optibet koeficientu vērtēšana)
- `bot.py` — LIKMES / REZULTĀTI (+ kopīgā vēsture/eksports, `League`), `backtest.py` — kalibrēšana
- NHL: `nhl_data.py`, `nhl_model.py` (Puasona vārtu režģis), `nhl_bets.py`, `nhl_bot.py`, `nhl_backtest.py`
- EuroLeague: `el_bot.py` (oficiālais EuroLeague avots, bez koeficientiem, `el_params.json`), lapa `site/el/`
- Čempionu līga: `cl_bot.py` (ESPN + koeficienti, 90 min. likmes, `cl_params.json`), lapa `site/cl/`
- Drošās likmes lapa `site/drosas/`: katrs robots saglabā `safe_all` (katrā spēlē drošākā likme ar koef. ≥ 1.7,
  `bot.safe_all` / `choose_safe(all_games=True)`), `site.py` `safe_page_data()` savāc visas līgas; sava statistika.
- Eiropas hokejs: `euhockey.py` + `cz_bot.py` (Čehijas Extraliga, hokej.cz HTML), `del_bot.py` (DEL, penny-del.org mēnešu
  tabulas), `shl_bot.py` (SHL, shl.se API `sports-v2/game-schedule`). Bez koeficientiem: robota Puasona modelis (uzbrukums/
  aizsardzība + mājas laukums, `<liga>_params.json`, `python cz_bot.py kalibret`), likmes kā NHL (pamatlaiks), lapas `site/cz|del|shl/`,
  likmes ar `likmes_eu` (12:30), iekļautas kopējā statistikā un 1.7+ lapā.
- Bukmeikeru līnijas SHL un EuroLeague: `odds.py` (The Odds API, `ODDS_API_KEY` = GitHub secret; Pinnacle vai citu EU
  bukmeikeru mediāna; kešs 2 h; zem 40 kredītiem neprasa). SHL kā NHL (80 % tirgus / 20 % modelis, ★/1.7+ pēc tirgus),
  EuroLeague kā NBA (tirgus handikaps/kopā). Bez atslēgas — robota modelis.
- Izņemti 26.11.2026: Championship, lapas 1.5+, Papildu un Robots (kods git vēsturē). NCAA izņemts 21.11.2026.
- Eiropas līgu likmes palaiž `likmes_eu` (12:17 Rīgā); rezultāti — visām līgām kopā no rīta.
  Rokas palaišana vienai līgai: Actions → Run workflow → `rezultati_nba|nhl|el|cl|cz|del|shl`.
- Kopējā statistika: `site/stats/` (`site.py` `combined_data()` + `renderStatsPage()` veidnē) — visas līgas
  kopā, 4 grupas (visas, ★, drošās, Crazy Dazy) pa nedēļām (P–Sv), mēnešiem, gadiem.
- Simulācijas atziņa (19.11.2026): robota paša modeļa novirze no bukmeikeriem pasliktināja izvēli;
  ★ un drošās likmes izvēlas pēc bukmeikeru varbūtības, NBA `market_weight_suggested` = 1.0.
