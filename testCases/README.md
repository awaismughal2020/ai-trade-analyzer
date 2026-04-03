# testCases

Run `/predict` (and future endpoints) against lists of tokens and save **Payload + Response** to a single JSON file.

## Quick start

From the **repo root**:

```bash
# Ensure the API is running (e.g. coinMarketAnalyzer on port 8000), then:
python run_predict_tests.py
```

This uses `testCases/inputs/meme_mints.txt` and writes to `testCases/results/predict_meme_<timestamp>.json`.

## Input files (edit when needed)

| File | Purpose |
|------|--------|
| `inputs/meme_mints.txt` | Meme token mints (one per line). Used by `--test-type meme`. |
| `inputs/perps_symbols.txt` | Perps symbols (e.g. BTC-USD). Used by `--test-type perps`. |
| `inputs/meme_wallets.txt` | Wallet addresses for future wallet/user tests. |

Lines starting with `#` and blank lines are ignored.

## Options

- `--test-type meme|perps|meme_wallets` – Which input file and `token_type` to use (default: `meme`).
- `--input-file PATH` – Override input file (e.g. a custom list).
- `--base-url URL` – API base URL (default: `http://localhost:8000`).
- `--output PATH` – Output JSON path (default: `results/predict_<test_type>_<timestamp>.json`).
- `--timeout SECS` – Request timeout (default: 120).

## Output format

Single JSON file with:

- **test_type**, **base_url**, **timestamp**, **count**
- **results**: array of `{ "payload": { ... }, "response": { ... } }` (one per token)

## Extending (perps, meme wallets)

- **Perps**: Add symbols to `inputs/perps_symbols.txt` and run  
  `python run_predict_tests.py --test-type perps`
- **Meme wallets**: Add addresses to `inputs/meme_wallets.txt`. The script already supports `--test-type meme_wallets`; you can later add a separate path that calls `/predict/with-user` or another endpoint by extending `run_predict_tests.py`.
