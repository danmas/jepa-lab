# База знаний проекта «jepa-lab»

**Актуализация:** 2026-11-12

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
| [README_report.md](README_report.md) | **Отчёт для руководства** — без техники: что и как сделано |
| [README_report_2.md](README_report_2.md) | **Отчёт 2 для руководства** — четыре усложнения полигона, без техники |
| [README_report_3.md](README_report_3.md) | **Отчёт 3 для руководства** — на странице виден шаг времени, без техники |
| [README_DT_DEMO.md](README_DT_DEMO.md) | Техническая заметка показа: одна сеть с явным `dt`, не замер |
| [README_WALLS.md](README_WALLS.md) | Стенки как параметр мира: 24 сети, раскатка, без победителя |
| [README_WORLDS.md](README_WORLDS.md) | Четыре мира v5: пробуксовка, помехи, два мяча, бугры. Ярлыки на 90 эпизодах |

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
| Что показать начальству | [README_report.md](README_report.md) — первый отчёт; [README_report_2.md](README_report_2.md) — четыре усложнения; [README_report_3.md](README_report_3.md) — шаг времени на странице |
| Где описан показ сети с явным dt | [README_DT_DEMO.md](README_DT_DEMO.md) |
| Где стенки вынесены в параметр мира | [README_WALLS.md](README_WALLS.md) |
| Где доля поля сравнивается с абсолютным масштабом | [README_SCALE.md](README_SCALE.md) |
| Где посадка учится без формулы отскока | [README_LAND.md](README_LAND.md) |
| Где большое поле посадки и бак 10 | [README_LAND_FIELD.md](README_LAND_FIELD.md) |
| Где план держится на пробуксовке, помехах, втором мяче и буграх | [README_WORLDS.md](README_WORLDS.md) |

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
| `demo_ui.py` | страница проверки на `:8765`, POST `/api/run`; модели `pixel`, `numeric`, `numeric-dt`, `numeric-walls` |
| `numeric_dt/` | интегратор и JEPA с явным `dt`; на странице обучена только рука B, сид 0 |
| `numeric_walls/` | JEPA со стенками как параметром мира; на странице протокол диапазона, сид 0 |
| `_check_numeric_dt.py` | контрольные сдвиги нового интегратора |
| `imagine.py` | раскатка архивного эксперимента v0.2 |
| `AGENTS.md` | правила работы ИИ-агентов в проекте |
| `CHANGELOG.md` | история изменений |
| `out/` | чекпоинты, логи, `eval_v4_*.json`, `eval_v5-*_latent.json` |
| `docs/plans/` | проектный документ v0.3 |
| `README.md` | описание для людей и таблица результатов |

**Последнее обновление:** 2026-11-12
