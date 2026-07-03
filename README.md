# PhD Integrator Project

**TP2 — Du Calcul Symbolique à l'IA Scientifique Multi-GPU avec Automatisation de la
Reproductibilité (CI/CD)**
Formation Doctorale : Calcul Scientifique, HPC et Génie Logiciel Avancé — Cycle Ph.D. D1

[![Scientific Computing CI/CD Pipeline](https://github.com/medbeddi/phd_integrator_project/actions/workflows/ci_cd_pipeline.yml/badge.svg)](https://github.com/medbeddi/phd_integrator_project/actions/workflows/ci_cd_pipeline.yml)

---

## 1. Vue d'ensemble

Ce dépôt implémente un pipeline de recherche reproductible qui simule un phénomène de
transfert de masse régi par l'équation d'advection-diffusion :

```
∂u/∂t + c·∂u/∂x − ν·∂²u/∂x² = f(x, t)
```

La solution analytique candidate étudiée est l'onde solitaire `u(x,t) = tanh(x − c·t)`
(méthode des solutions manufacturées). Le pipeline enchaîne :

1. **Dérivation symbolique** (SymPy) du terme source résiduel exact `f(x,t)`.
2. **Ingestion et vectorisation** (Polars Lazy API + NumPy broadcasting) sur une grille
   spatio-temporelle.
3. **Analyse de stabilité et de conditionnement** (matrice de Hilbert, IEEE 754,
   float16/32/64).
4. **Accélération HPC** (Numba `@njit(parallel=True, fastmath=True)` + parallélisme
   multi-processus).
5. **Approximation par PINN** (Physics-Informed Neural Network, PyTorch, autograd).
6. **Visualisation scientifique** (Matplotlib/Seaborn statique PDF, Plotly interactif HTML).
7. **Orchestration reproductible** via Snakemake (DAG) et **CI/CD** via GitHub Actions.

---

## 2. Procédure exacte de déploiement (uv)

Le projet est géré exclusivement par [`uv`](https://docs.astral.sh/uv/), sans installation
globale ni privilège root (contrainte *No-Sudo*, reproductibilité de type supercalculateur).

```bash
# 1. Installer uv (utilisateur, sans sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Installer l'interpréteur Python géré par uv (isolé, espace utilisateur)
uv python install 3.11

# 3. Cloner puis synchroniser l'environnement virtuel isolé + verrouillage déterministe
git clone <URL_DU_DEPOT>
cd phd_integrator_project
uv sync --python 3.11          # crée .venv/ et installe depuis uv.lock

# 4. Validation locale complète (identique à la CI)
uv run ruff check src/ tests/ scripts/
uv run mypy --strict src/
uv run pytest --cov=src --cov-report=term-missing tests/

# 5. Exécution du pipeline scientifique complet (DAG Snakemake)
uv run snakemake --cores 1 -p
```

Les artéfacts sont produits dans `outputs/` :
`outputs/figures/solution_and_error.pdf`, `outputs/figures/interactive_surface.html`,
`outputs/models/pinn_weights.pt`, `outputs/reports/stability_report.txt`.

### Pourquoi `pulp<2.8` en plus des dépendances demandées ?

`snakemake` 8+ exige Python ≥ 3.11, ce qui casserait la matrice CI 3.10/3.11/3.12 imposée
par l'énoncé. Le projet épingle donc `snakemake>=7.32,<8` (dernière branche compatible
3.10-3.12). Cette branche appelle en interne `pulp.list_solvers`, une API supprimée dans
`pulp` 3.x — d'où l'épinglage `pulp<2.8` pour garantir un `uv sync` déterministe sur les
trois versions cibles. Décision documentée ici plutôt que masquée, conformément à
l'exigence de rigueur scientifique du cahier des charges.

---

## 3. Structure du dépôt

```
phd_integrator_project/
├── .github/workflows/ci_cd_pipeline.yml   # CI/CD (matrix 3.10/3.11/3.12, artefacts)
├── data/raw_sensors/                      # Données climatiques synthétiques (Parquet)
├── src/
│   ├── symbolic_derivation.py             # Module 3 — SymPy
│   ├── numerical_core.py                  # Module 4 — NumPy / Polars
│   ├── stability_analysis.py              # Module 5 — Conditionnement / IEEE 754
│   ├── hpc_acceleration.py                # Module 6 — Numba JIT / multiprocessing
│   ├── deep_pinn.py                       # Module 7 — PINN PyTorch
│   └── visualization.py                   # Module 8 — Matplotlib/Seaborn/Plotly
├── scripts/                               # Points d'entrée orchestrés par Snakemake/CI
├── tests/                                 # Suite PyTest (95% de couverture)
├── outputs/{figures,models,reports}/      # Artefacts générés (régénérés par la CI)
├── pyproject.toml                         # Dépendances (PEP 518) + config ruff/mypy/pytest
├── uv.lock                                # Verrouillage déterministe
└── Snakefile                              # Orchestration DAG
```

---

## 4. Explication physique et mathématique

Pour `u(x,t) = tanh(x − c·t)` :

- `∂u/∂t = −c·(1 − u²)`
- `∂u/∂x = (1 − u²)`
- `∂²u/∂x² = −2u·(1 − u²)`

En injectant dans l'opérateur d'advection-diffusion :

```
f(x,t) = ∂u/∂t + c·∂u/∂x − ν·∂²u/∂x² = 2·ν·u·(1 − u²)
```

Le terme d'advection s'annule identiquement (`−c(1−u²) + c(1−u²) = 0`) : l'onde solitaire
est advectée sans déformation. Le résidu non nul provient uniquement du terme diffusif,
proportionnel à `ν`. C'est cette expression exacte (calculée symboliquement au Module 3,
puis lambdifiée) qui sert de **terme source manufacturé** dans la perte physique du PINN
(`src/deep_pinn.py::physics_residual`), permettant d'entraîner le réseau sans jeu de
données de référence externe — le résidu de l'EDP est calculé par différentiation
automatique (`torch.autograd.grad`, ordre 2) et comparé analytiquement à `f(x,t)`.

---

## 5. Analyse critique : précision flottante et accélérations logicielles

### Conditionnement et précision (Module 5)

La matrice de Hilbert `A_ij = 1/(i+j+1)` est un cas d'école de mauvais conditionnement :
`κ(A)` croît exponentiellement avec `n` (empiriquement `κ(A) ~ 10^(1.5n)` au-delà de
`n ≈ 12`). Les résultats (`outputs/reports/stability_report.txt`) montrent que :

- En **float16**, l'erreur de reconstruction explose dès `n ≈ 8-10` (mantisse de 10 bits
  insuffisante face à `κ(A)` qui dépasse rapidement `10^13`).
- En **float32**, la dégradation apparaît vers `n ≈ 12-14`.
- En **float64**, la précision reste acceptable jusqu'à `n ≈ 15-16`, avant que même la
  double précision ne soit dominée par `κ(A) ≈ 1/ε_machine`.

Une perturbation `ε = 10⁻⁷` sur le second membre `b` est amplifiée sur la solution `α`
dans un rapport proche de `κ(A)` (borne théorique : `‖δα‖/‖α‖ ≤ κ(A)·‖δb‖/‖b‖`), validée
empiriquement par `stability_analysis.perturbation_sensitivity`. C'est pourquoi toute
validation numérique du projet utilise `np.isclose(atol=..., rtol=...)` et jamais `==` :
l'égalité stricte sur des flottants ayant transité par une résolution linéaire mal
conditionnée échouerait presque systématiquement, sans indiquer une réelle erreur du
programme — seulement l'absence de représentation exacte en base 2 (IEEE 754).

### Numba JIT et `fastmath=True` (Module 6)

`@njit(parallel=True, fastmath=True)` apporte un gain mesurable (`benchmark_filter`) en
compilant les boucles imbriquées en code machine natif et en distribuant l'axe externe sur
les cœurs disponibles (`prange`). `fastmath=True` autorise le compilateur LLVM à
réordonner les opérations flottantes (associativité supposée, gestion relâchée de NaN/Inf),
ce qui viole formellement l'associativité stricte imposée par IEEE 754 : les résultats
peuvent différer légèrement (dernier bit de mantisse) selon l'ordre de réduction choisi par
le compilateur. Ce compromis est acceptable pour un filtre de lissage (tolérance
`atol=1e-5` dans les tests), mais serait risqué pour un solveur itératif sensible aux
erreurs d'arrondi accumulées (cf. Module 5).

### PINN et choix de précision (Module 7)

L'entraînement du PINN utilise `float32` (par défaut PyTorch) pour tirer parti de
l'accélération matérielle (Tensor Cores CUDA/MPS), au prix d'une précision moindre que
`float64` sur le résidu physique — acceptable ici car la perte est elle-même une moyenne
quadratique sur un batch de points de collocation, ce qui lisse le bruit d'arrondi
individuel. Le code détecte automatiquement le meilleur device disponible
(`get_device()` : CUDA > MPS > CPU) pour permettre le passage à l'échelle sans
modification du code applicatif ; sur un cluster HPC réel, la stratégie de mise à l'échelle
naturelle serait le parallélisme de données (`torch.nn.parallel.DistributedDataParallel`)
combiné à des solveurs linéaires HPC sous-jacents (PETSc) pour les composantes non-DL du
pipeline (assemblage de `A·α = b` à grande échelle).

---

## 6. Rendu LaTeX des figures et contrainte No-Sudo

L'énoncé impose `text.usetex: True`. Une installation TeX Live complète via `apt-get`
nécessite cependant des privilèges root sur les exécuteurs GitHub Actions, ce qui entre en
conflit direct avec la contrainte *Reproductibilité Absolue (No-Sudo)* du cahier des
charges. `src/visualization.py::configure_latex_rendering()` active donc le rendu LaTeX
natif **si et seulement si** un binaire `latex` est détecté sur le `PATH` (poste de
recherche local avec MiKTeX/TeX Live), et se replie automatiquement sur le moteur
`mathtext` interne de Matplotlib sinon — qui produit un rendu mathématique quasi-identique
sans dépendance externe. Ce choix explicite privilégie la contrainte de sécurité/
reproductibilité (prioritaire dans le cadre institutionnel) sur le rendu typographique
strict, tout en conservant l'intégralité du pipeline exécutable sans intervention manuelle.

---

## 7. CI/CD : déclencheurs, matrice et artefacts

Le workflow `.github/workflows/ci_cd_pipeline.yml` se déclenche sur `push` et
`pull_request` vers `main`, avec une **matrice** testant Python 3.10, 3.11 et 3.12 en
parallèle :

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
```

Chaque job effectue : checkout → installation `uv` (cache activé) → `uv sync --locked` →
`ruff check` → `mypy --strict` → `pytest --cov`. Sur le job Python 3.11, une étape
supplémentaire régénère les artefacts scientifiques (figures PDF/HTML, poids du modèle) et
les publie via `actions/upload-artifact`, téléchargeables depuis l'onglet *Actions* de
GitHub, uniquement si toutes les étapes précédentes ont réussi (`if: success()`).

**Utilité de `enable-cache: true`** : il persiste le cache de résolution et les
téléchargements `uv` (wheels, notamment `torch` ≈ 2-3 Go) entre exécutions du workflow.
Sans cache, chaque run retélécharge l'intégralité des dépendances depuis PyPI, ce qui
multiplie le temps d'exécution des runners (donc leur empreinte carbone, proportionnelle au
temps CPU consommé sur l'infrastructure partagée GitHub) et la bande passante réseau
consommée à chaque déclenchement — un coût non négligeable à l'échelle d'un dépôt de
recherche avec des dizaines de push quotidiens.

**Utilité de `mypy --strict`** : sur des simulations HPC de plusieurs heures, une erreur de
typage (ex. passer une `str` là où un `np.ndarray` est attendu) ne se manifeste souvent
qu'au moment de l'exécution effective — après un temps de calcul potentiellement très
coûteux sur un supercalculateur. La vérification statique détecte ces incohérences
*avant* toute exécution, à coût quasi nul, évitant de gaspiller des heures de calcul
allouées sur des files d'attente partagées à cause d'une erreur détectable en quelques
secondes.

---

## 8. Validation expérimentale du pipeline CI (défi Exercice 9.2)

L'historique Git de ce dépôt documente le scénario de défaillance volontaire demandé par
l'énoncé :

1. Un commit introduit une erreur de typage flagrante (passage d'une `str` à une fonction
   attendant un `np.ndarray`), provoquant l'échec de l'étape `mypy --strict` de la CI.
2. Le commit suivant corrige l'erreur et ajoute un test de régression dédié
   (`test_numerical_core_rejects_wrong_type_via_mypy` — cas limite documenté), avec un
   message respectant les *Conventional Commits* (`fix: ...`).

Voir `git log --oneline` pour l'enchaînement exact des commits correspondants.

---

## 9. Badges et artefacts

- **Badge de statut** : voir l'en-tête de ce document (mis à jour automatiquement par
  GitHub après le premier push sur `main`).
- **Historique des déclenchements** : onglet *Actions* du dépôt GitHub.
- **Artefacts téléchargeables** : à la fin de chaque run réussi sur Python 3.11, dans
  *Actions → (run) → Artifacts → scientific-outputs* (figures PDF/HTML + poids `.pt`).
