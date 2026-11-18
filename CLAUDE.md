# NBA likmju robots — instrukcijas Claude

Lietotājs raksta latviski. Atbildi latviski.

## Komandas čatā

- **LIKMES** (jebkurā reģistrā) → palaid `python3 bot.py likmes --refresh` šajā mapē un parādi
  rezultātu: katrai šīs nakts spēlei 3 likmes ar koeficientu ≥ 2.1 un "drošību" 1–100.
  Ja lietotājs norāda citu min. koeficientu, pievieno `--min-odds X`.
  Pēc tam īsi (2–4 teikumos) izcel interesantāko: spēles ar lieliem savainojumiem vai kur
  robots būtiski nepiekrīt bukmeikeriem. Neizdomā varbūtības — izmanto tikai skripta skaitļus.
- **REZULTĀTI** → palaid `python3 bot.py rezultati` un parādi kopsavilkumu (kuras likmes
  izgāja, dienas un kopējā statistika).
- Likmju vēsture glabājas `history/*.json`. Pēc LIKMES un REZULTĀTI saglabā izmaiņas
  (git commit + push), citādi vēsture pazudīs līdz ar sesiju.

## Automātika

LIKMES (21:30) un REZULTĀTI (08:30, Rīgas laiks) automātiski palaiž GitHub Actions
(`.github/workflows/nba-prognozes.yml` + `scheduler.py`). Pēc tam tiek publicēta publiskā lapa
https://edzinslabakais.github.io/nba-prognozes/ (`site.py` → `site/`, GitHub Pages), kā arī atjaunināts
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
- `bot.py` — LIKMES / REZULTĀTI, `backtest.py` — precizitātes pārbaude un `params.json` kalibrēšana
