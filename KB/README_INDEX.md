# База знаний проекта «jepa-lab»

**Актуализация:** 2026-11-10

Учебная лаборатория JEPA на CPU. Два эксперимента в одном репозитории:

1. **«Воображение»** (v0.2, архив) — модель катит мир вперёд. Числа в `out/metrics.json`.
2. **«Цель словами»** (v0.3, текущий код) — фраза → план в латенте → мяч в зоне.
   Чекпоинты `out/jepa_v4.pt` и `out/pixel_v4.pt`.

Описание для чтения с нуля — корневой [README.md](../README.md). Здесь те же факты,
разложенные по файлам, и они совпадают с кодом.

## Быстрый старт

```bash
pip install -r requirements.txt
python train.py --mode jepa  --steps 2500 --tag v4
python train.py --mode pixel --steps 2500 --tag v4
python probe.py --tag v4
python chat.py --tag v4
```

Пустой `--tag` у `train.py` пишет `out/jepa.pt` / `out/pixel.pt` и затирает архив v0.2.

## Навигация

### 1. Архитектура и теория

| Файл | О чём |
|------|-------|
| [README_ARCH.md](README_ARCH.md) | Компоненты, лосс v0.3, физика, архивная раскатка v0.2 |
| [README_PLAN.md](README_PLAN.md) | Цель, CEM, MPC, два режима стоимости, три грабли v0.3 |
| [README_DEV.md](README_DEV.md) | Команды, флаги, времена, константы физики |

### 2. Быстрый поиск по темам

| Вопрос | Куда смотреть |
|--------|---------------|
| Как устроен энкодер/декодер/предиктор | [README_ARCH.md](README_ARCH.md) → «Компоненты» |
| Чем JEPA-режим отличается от pixel-режима | [README_ARCH.md](README_ARCH.md) → «Два режима — один код» |
| Какие сейчас гравитация, импульс, затухание | [README_ARCH.md](README_ARCH.md) → «Физика» |
| Как задаётся цель и как ищется план | [README_PLAN.md](README_PLAN.md) → «Цель», «Поиск» |
| Чем `latent` отличается от `mixed` | [README_PLAN.md](README_PLAN.md) → «Два режима стоимости» |
| Где в этой схеме LLM | [README_PLAN.md](README_PLAN.md) → «Три слоя» |
| Почему планировщик сначала не работал | [README_PLAN.md](README_PLAN.md) → «Что пришлось чинить» |
| Откуда таблица дрейфа 0.8 px против 6.2 px | [README_ARCH.md](README_ARCH.md) → «Раскатка v0.2» |
| Сколько это учится на CPU | [README_DEV.md](README_DEV.md) → «Производительность» |
| Где посмотреть мяч глазами | `python demo_ui.py` → http://127.0.0.1:8765 |
| Как устроены спеки изменений | [README_DEV.md](README_DEV.md) → «Спеки (OpenSpec)» |
| Какой командой сняты 90 эпизодов | [README_DEV.md](README_DEV.md) → «Запуск» |
| Как поменять физику мяча | [README_DEV.md](README_DEV.md) → «Эксперименты» |

## Состав репозитория

| Файл | Назначение |
|------|-----------|
| `data.py` | физика (`GRAVITY` 0.15, `IMPULSE` 4.0, `VEL_DECAY` 0.6), зоны, `World` |
| `models.py` | `Encoder`, `Decoder`, `Predictor`, `EMATarget`, `Probe` |
| `train.py` | обучение `jepa` / `pixel`, `HORIZON = 6`, три шага раскатки |
| `probe.py` | ridge: состояние мира из пары латентов |
| `plan.py` | CEM в латенте + MPC, `--cost latent\|mixed` |
| `eval.py` | успех, время, корреляция, сетка зон, пиксельный baseline |
| `chat.py` | фраза → план → отчёт. Дефолт стоимости — `latent`, потолок 20 шагов |
| `imagine.py` | раскатка архивного эксперимента v0.2 |
| `out/` | чекпоинты, логи, `eval_v4_latent.json`, `eval_v4_mixed.json` |
| `docs/plans/` | проектный документ v0.3 |
| `README.md` | описание для людей и таблица результатов |

**Последнее обновление:** 2026-11-10
