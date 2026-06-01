"""
analysis_semantic.py
--------------------
Семантическое сравнение предсказаний модели с ответами людей.

Эмбеддинги: ruwikiruscorpora_upos_skipgram_300_2_2019
  (статические word2vec, формат .bin)
  Слова закодированы в формате "лемма_UPOS", например "крупа_NOUN".

Метрики (для каждого контекста и каждого top-k):
  1. Centroid Similarity — косинусное сходство между взвешенным
     средним вектором человеческих ответов и взвешенным средним
     вектором топ-k предсказаний модели.
  2. NN Coverage @ threshold — для каждого человеческого ответа
     ищется ближайший сосед среди предсказаний модели; ответ
     засчитывается как покрытый, если cos-сходство ≥ θ.
     θ ∈ {0.70, 0.80, 0.90}

Оба показателя взвешены по вероятностям ответов / предсказаний.

Входные файлы:
  model.bin                  — ruwikiruscorpora KeyedVectors
  people_with_prob.csv       — human data (нужны lemma_answer + upos_answer)
  gpt4omini_morph_2.csv      — GPT predictions (нужны lemma_word + upos_word)
  llama_with_all.csv         — Llama predictions (нужны lemma_word + upos_word)

Выходные файлы → output/semantic/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import stats

from filter_data import load_human, get_all_models

OUT = Path("output/semantic")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

BLUE   = "#2B6CB0"
RED    = "#C53030"
ORANGE = "#C05621"
GRAY   = "#718096"

# ════════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ
# ════════════════════════════════════════════════════════════════════
WORD2VEC_PATH  = "model.bin"          # путь к ruwikiruscorpora .bin
TOP_K_VALUES   = [1, 5, 10, 20, 50, 100]
NN_THRESHOLDS  = [0.70, 0.80, 0.90]
EMB_DIM        = 300

# ════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ЭМБЕДДИНГОВ
# ════════════════════════════════════════════════════════════════════
print("Загружаем ruwikiruscorpora KeyedVectors...")
kv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
print(f"  Словарь: {len(kv)} токенов")


def get_vec(lemma: str, upos: str) -> np.ndarray:
    """
    Возвращает вектор для токена вида "лемма_UPOS".
    Если токен не найден, возвращает нулевой вектор.
    """
    key = f"{lemma.lower()}_{upos}"
    if key in kv:
        return kv[key].astype(float)
    # fallback: попробовать без UPOS
    if lemma.lower() in kv:
        return kv[lemma.lower()].astype(float)
    return np.zeros(EMB_DIM, dtype=float)


# ════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ════════════════════════════════════════════════════════════════════
people = load_human()

context_lookup = (
    people.drop_duplicates("word.id")[["word.id", "Left context"]]
    .rename(columns={"word.id": "target_word", "Left context": "left_context"})
)

# ── Human: дедупликация + эмбеддинги ────────────────────────────────
human_deduped = (
    people
    .assign(answer_stripped=people["answer"].str.strip())
    .drop_duplicates(subset=["word.id", "answer_stripped"])
    [["word.id", "answer_stripped", "lemma_answer", "upos_answer", "probability_y"]]
    .dropna(subset=["lemma_answer", "upos_answer"])
)

# предвычисляем все уникальные (лемма, UPOS) пары для human
human_keys = set(
    zip(human_deduped["lemma_answer"].str.lower(),
        human_deduped["upos_answer"].str.upper())
)

print(f"Human: {len(human_deduped)} записей, {len(human_keys)} уникальных лемм×UPOS")

# ════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ════════════════════════════════════════════════════════════════════

def weighted_centroid(vecs: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Взвешенное среднее векторов, нормализованное до единичной длины."""
    centroid = (probs[:, None] * vecs).sum(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / (norm + 1e-9)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица косинусных сходств (a уже нормализован в kv)."""
    # векторы из kv уже нормализованы — просто dot product
    return a @ b.T


def nn_coverage(h_probs: np.ndarray, sim_matrix: np.ndarray,
                threshold: float) -> float:
    """
    NN Coverage @ threshold.
    Для каждого человеческого ответа i берём максимальное
    косинусное сходство с предсказаниями модели.
    Ответ «покрыт», если max_sim >= threshold.
    Coverage = сумма вероятностей покрытых ответов.
    """
    best_sim = sim_matrix.max(axis=1)           # (n_human,)
    covered  = (best_sim >= threshold).astype(float)
    return float(np.dot(h_probs, covered))


def centroid_similarity(h_probs: np.ndarray, h_vecs: np.ndarray,
                        m_probs: np.ndarray, m_vecs: np.ndarray) -> float:
    """
    Cosine similarity между взвешенными центроидами
    человеческого и модельного облаков.
    """
    h_cent = weighted_centroid(h_vecs, h_probs)
    m_cent = weighted_centroid(m_vecs, m_probs)
    return float(np.dot(h_cent, m_cent))


# ════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ЦИКЛ: по моделям
# ════════════════════════════════════════════════════════════════════

all_results = {}

for model_name, model_df in get_all_models().items():
    print(f"\n{'=' * 55}")
    print(f"  Модель: {model_name}")
    print(f"{'=' * 55}")

    model_slug = model_name.replace(" ", "_").replace("-", "_")

    # дедупликация предсказаний по поверхностной форме
    model_sorted = model_df.sort_values("probability_converted", ascending=False)
    model_deduped = (
        model_sorted
        .drop_duplicates(subset=["target_word_id", "pred_stripped"], keep="first")
        .dropna(subset=["lemma_word", "upos_word"])
        [["target_word_id", "pred_stripped", "lemma_word",
          "upos_word", "probability_converted"]]
    )

    # предвычисляем все уникальные (лемма, UPOS) пары для модели
    model_keys = set(
        zip(model_deduped["lemma_word"].str.lower(),
            model_deduped["upos_word"].str.upper())
    )

    # строим кэш векторов
    all_keys = human_keys | model_keys
    vec_cache = {(lem, upos): get_vec(lem, upos) for lem, upos in all_keys}

    n_found = sum(
        1 for v in vec_cache.values() if v.sum() != 0
    )
    print(f"  Векторов найдено: {n_found}/{len(all_keys)} "
          f"({100*n_found/max(len(all_keys),1):.1f}%)")

    common_ids = sorted(
        set(human_deduped["word.id"]) & set(model_deduped["target_word_id"])
    )
    print(f"  Общих контекстов: {len(common_ids)}")

    records = []

    for word_id in common_ids:
        h_sub = human_deduped[human_deduped["word.id"] == word_id]
        m_sub = model_deduped[model_deduped["target_word_id"] == word_id]

        if len(h_sub) == 0 or len(m_sub) == 0:
            continue

        # ── Human vectors ────────────────────────────────────────────
        h_vecs = np.array([
            vec_cache.get(
                (row["lemma_answer"].lower(), row["upos_answer"].upper()),
                np.zeros(EMB_DIM)
            )
            for _, row in h_sub.iterrows()
        ])
        h_probs = h_sub["probability_y"].values.astype(float)
        h_probs = h_probs / (h_probs.sum() + 1e-9)

        # фильтруем нулевые векторы (слово не в словаре)
        nonzero_h = h_vecs.sum(axis=1) != 0
        if nonzero_h.sum() == 0:
            continue
        h_vecs  = h_vecs[nonzero_h]
        h_probs = h_probs[nonzero_h]
        h_probs = h_probs / (h_probs.sum() + 1e-9)

        # ── Model vectors (все, потом режем до top-k) ─────────────────
        m_sub_sorted = m_sub.sort_values(
            "probability_converted", ascending=False
        ).reset_index(drop=True)

        m_vecs_all = np.array([
            vec_cache.get(
                (row["lemma_word"].lower(), row["upos_word"].upper()),
                np.zeros(EMB_DIM)
            )
            for _, row in m_sub_sorted.iterrows()
        ])
        m_probs_all = m_sub_sorted["probability_converted"].values.astype(float)

        nonzero_m = m_vecs_all.sum(axis=1) != 0
        if nonzero_m.sum() == 0:
            continue
        m_vecs_all  = m_vecs_all[nonzero_m]
        m_probs_all = m_probs_all[nonzero_m]

        # ── Метрики для каждого top-k ─────────────────────────────────
        for k in TOP_K_VALUES:
            k_eff = min(k, len(m_vecs_all))
            m_vecs  = m_vecs_all[:k_eff]
            m_probs = m_probs_all[:k_eff].copy()
            m_probs = m_probs / (m_probs.sum() + 1e-9)

            sim_matrix = cosine_similarity_matrix(h_vecs, m_vecs)

            # 1. Centroid similarity
            c_sim = centroid_similarity(h_probs, h_vecs, m_probs, m_vecs)

            # 2. NN Coverage @ каждый порог
            rec = {
                "word_id":    word_id,
                "top_k":      k,
                "k_effective": k_eff,
                "centroid_sim": round(c_sim, 6),
            }
            for thr in NN_THRESHOLDS:
                cov = nn_coverage(h_probs, sim_matrix, thr)
                rec[f"nn_coverage_{int(thr*100)}"] = round(cov, 6)

            records.append(rec)

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUT / f"semantic_per_context_{model_slug}.csv", index=False)
    print(f"  Saved semantic_per_context_{model_slug}.csv "
          f"({len(results_df)} rows)")

    all_results[model_name] = results_df

    # ── Сводная таблица по top-k ──────────────────────────────────────
    summary_cols = ["centroid_sim"] + [
        f"nn_coverage_{int(t*100)}" for t in NN_THRESHOLDS
    ]
    summary = (
        results_df.groupby("top_k")[summary_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    summary.to_csv(OUT / f"semantic_summary_by_k_{model_slug}.csv", index=False)

    print(f"\n  Сводка по top-k [{model_name}]:")
    print("  " + summary.to_string(index=False))


# ════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ════════════════════════════════════════════════════════════════════

model_colors = {"GPT-4o-mini": RED, "Llama": ORANGE}

# ── 1. Centroid similarity vs top-k ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for model_name, df in all_results.items():
    summary = df.groupby("top_k")["centroid_sim"].mean().reset_index()
    color = model_colors.get(model_name, GRAY)
    ax.plot(summary["top_k"], summary["centroid_sim"],
            "o-", color=color, linewidth=2, markersize=6,
            label=model_name)
    for _, row in summary.iterrows():
        ax.annotate(f"{row['centroid_sim']:.3f}",
                    (row["top_k"], row["centroid_sim"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=color)

ax.set_xlabel("top-k предсказаний", fontsize=11)
ax.set_ylabel("Centroid Similarity (cosine)", fontsize=11)
ax.set_title("Семантическое сходство центроидов: Human vs. Модель",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig_centroid_by_k.png", bbox_inches="tight")
plt.close()
print("\nSaved fig_centroid_by_k.png")

# ── 2. NN Coverage @ threshold vs top-k ─────────────────────────────
for thr in NN_THRESHOLDS:
    col = f"nn_coverage_{int(thr*100)}"
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, df in all_results.items():
        if col not in df.columns:
            continue
        summary = df.groupby("top_k")[col].mean().reset_index()
        color = model_colors.get(model_name, GRAY)
        ax.plot(summary["top_k"], summary[col],
                "s--", color=color, linewidth=2, markersize=6,
                label=model_name)
        for _, row in summary.iterrows():
            ax.annotate(f"{row[col]:.3f}",
                        (row["top_k"], row[col]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=color)

    ax.set_xlabel("top-k предсказаний", fontsize=11)
    ax.set_ylabel(f"NN Coverage @ {thr}", fontsize=11)
    ax.set_title(f"Семантическое покрытие человеческих ответов (θ = {thr})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fname = f"fig_nn_coverage_{int(thr*100)}_by_k.png"
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# ── 3. Scatter: centroid_sim vs H_human (если энтропия посчитана) ────
entropy_path = Path("output/entropy/per_context_entropy_human.csv")
if entropy_path.exists():
    entropy_df = pd.read_csv(entropy_path)

    for model_name, df in all_results.items():
        # берём только k=100 для иллюстрации
        df_k = df[df["top_k"] == 100].copy()
        merged = df_k.merge(
            entropy_df[["target_word", "H_human"]],
            left_on="word_id", right_on="target_word", how="inner"
        )
        if len(merged) < 10:
            continue

        r, p = stats.pearsonr(merged["H_human"], merged["centroid_sim"])
        color = model_colors.get(model_name, GRAY)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(merged["H_human"], merged["centroid_sim"],
                   alpha=0.55, s=30, color=color, zorder=3)
        z = np.polyfit(merged["H_human"], merged["centroid_sim"], 1)
        x_line = np.linspace(merged["H_human"].min(),
                             merged["H_human"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="black",
                linewidth=1.5, linestyle="--")
        ax.set_xlabel("H_human (энтропия контекста)", fontsize=11)
        ax.set_ylabel("Centroid Similarity (top-100)", fontsize=11)
        ax.set_title(f"{model_name}\n"
                     f"r = {r:.3f}, p = {p:.4f}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        slug = model_name.replace(" ", "_").replace("-", "_")
        plt.savefig(OUT / f"fig_entropy_vs_centroid_{slug}.png",
                    bbox_inches="tight")
        plt.close()
        print(f"Saved fig_entropy_vs_centroid_{slug}.png")

print(f"\n✓ Готово. Файлы → {OUT}/")
