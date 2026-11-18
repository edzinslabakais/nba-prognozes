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

LIKMES (21:30), REZULTĀTI (07:00 un vēlreiz 08:30) un WhatsApp statistiku (07:30, `whatsapp.py`, CallMeBot;
Rīgas laiks) automātiski palaiž GitHub Actions
(`.github/workflows/nba-prognozes.yml` + `scheduler.py`). Pēc tam tiek publicēta publiskā lapa
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

## Struktūra

- `espn.py` — ESPN API + kešs (`cache/`), `data.py` — tabulas, `model.py` — reitingi un savainojumi
- `predict.py` — spēļu prognozes, `bets.py` — likmju varbūtības (cenu lapa / Optibet koeficientu vērtēšana)
- `bot.py` — LIKMES / REZULTĀTI (+ kopīgā vēsture/eksports, `League`), `backtest.py` — kalibrēšana
- `archive.py` — simulēts arhīvs (`archive/`, `nhl/archive/`, "simulated": true; neskaitās īstajā statistikā)
- NHL: `nhl_data.py`, `nhl_model.py` (Puasona vārtu režģis), `nhl_bets.py`, `nhl_bot.py`, `nhl_backtest.py`
