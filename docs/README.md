# docs

Документация и интерактивные canvas для двух тем:

- **Diff Safetensors** — как из `feature_stories` получается `diff.safetensors`
- **Korznikov Dataset** — обзор пар и историй из `AntonKorznikov/feature_stories`

## Canvases

| Файл | Тема |
|------|------|
| `canvases/diff-safetensors.canvas.tsx` | Пошаговый пайплайн diff |
| `canvases/korznikov-dataset.canvas.tsx` | Каталог классов и историй (embedded data) |

Откройте `.canvas.tsx` в Cursor рядом с чатом.

## Просмотр в браузере (без Cursor)

```bash
cd docs/viewer
npm install
npm run dev
```

Откройте http://localhost:5173 — два canvas в табах.

Статическая сборка:

```bash
npm run build
npm run preview
```

### Доступ для людей в той же сети (машина airi)

```bash
cd docs/viewer
npm run build          # один раз после изменений
./serve-lan.sh         # или: npm run preview:lan
```

Коллеги открывают в браузере **`http://<IP-машины-airi>:4173/`**  
IP на airi: `hostname -I` (первая колонка).

Сервер слушает все интерфейсы (`0.0.0.0`), не только localhost.  
Если не открывается — проверьте firewall на airi (порт **4173**).

Viewer подменяет `cursor/canvas` локальным shim (`viewer/src/canvas/`) и импортирует те же файлы из `docs/canvases/`.

## Korznikov canvas — обновление данных

```bash
cd docs/scripts

# 1. Скачать истории одного класса с HF (долго, streaming)
python export_korznikov_stories_bundle.py --class-name "Vulnerability & Resilience"

# 2. Вшить bundle в canvas (+ опционально sync в Cursor canvases/)
python embed_korznikov_canvas.py
```

`export_korznikov_stories_bundle.py` читает pair index из `01_eval/results/concept_viewer/index.json`
(или свой `--index`). Diff / concept-vectors canvases статичны, скрипты не нужны.

## Зависимости

```bash
pip install datasets huggingface-hub
```

(GPU не требуется — только export/embed.)
