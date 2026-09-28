<h1 align="center">
Marvelous MLOps Free End-to-end MLOps with Databricks Course

## Set up your environment
In this course, we use Databricks serverless [version 3](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/three)

In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```



# Data
Using the [**Marvel Characters Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download) from Kaggle.

This dataset contains detailed information about Marvel characters (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.

# Scripts

- `01.process_data.py`: Loads and preprocesses the Marvel dataset, splits into train/test, and saves to the catalog.
- `02.train_register_fe_model.py`: Performs feature engineering and trains the Marvel character model.
- `03.deploy_model.py`: Deploys the trained Marvel model to a Databricks model serving endpoint.
- `04.post_commit_status.py`: Posts status updates for Marvel integration tests to GitHub.
- `05.refresh_monitor.py`: Refreshes monitoring tables and dashboards for Marvel model serving.

# Fixes and changes log

Problems found while running this project locally on Windows, what they caused, and how they were fixed.

## 1. Notebooks could not get a Spark session locally (`SparkSession` → `DatabricksSession`)

- **Problem:** the notebooks were written for the Databricks workspace and create Spark with `SparkSession.builder.getOrCreate()`. Run locally from VS Code, there is no Spark cluster behind that call, so the notebooks could not read the Unity Catalog tables (`train_set`, `test_set`).
- **Error:** the Spark cells did not run locally (no session to the workspace).
- **Fix:** switched to Databricks Connect, which runs the Spark code on the workspace's serverless compute:
  ```python
  from databricks.connect import DatabricksSession

  spark = DatabricksSession.builder.profile("dbc-07d12638-ef42").serverless(True).getOrCreate()
  ```
  Applied in `lecture2.marvel_data_preprocessing.py`, both `lecture4.*` notebooks and `lecture6.ab_testing.py`. `databricks-connect` 16.x (the `dev` extra) matches serverless environment 3.
- **Follow-up bug:** the new import in `lecture4.train_register_custom_model.py` was written as `from Databricks.connect import ...`. Python package names are case-sensitive, so it failed with `ModuleNotFoundError: No module named 'Databricks'`. Fixed to lowercase `databricks`.
- **Not yet converted:** `lecture6.deploy_model_serving_endpoint.py` and the first part of `lecture10.marvel_create_monitoring_table.py` still use `SparkSession`.

## 2. Custom model fails to load on the serving endpoint (Windows path bug)

- **Problem:** when a pyfunc model is logged with `artifacts=...`, MLflow (3.1.1, `mlflow/pyfunc/model.py`) records the artifact path with `os.path.join`. Logged from Windows, the `MLmodel` file stores `path: artifacts\.`. The Linux serving container joins that onto `/model/`, producing a path that does not exist.
- **Error:** the endpoint image builds, but the model never loads and the deployment ends in `DEPLOYMENT_FAILED` after ~10 minutes of retries:
  `OSError: No such file or directory: '/model/artifacts\.'`
- **Fix:**
  - `src/marvel_characters/models/custom_model.py`: `MarvelModelWrapper.load_context` normalizes the path with `.replace("\\", "/")` before `mlflow.sklearn.load_model`. Forward slashes also work on Windows, so local loading is unaffected.
  - Rebuilt the wheel (`uv build`) and re-registered the custom model. Each registered version carries the wheel it was logged with, so older versions keep the bug.
  - `notebooks/lecture6.deploy_model_serving_endpoint.py`: new **pre-deploy check** cell before the deploy cell. It loads the `latest-model` version, runs a prediction, then opens the wheel packaged inside the model and asserts the fix is present. A local prediction alone cannot catch this bug, because Windows accepts backslash paths.
- **Note:** models registered by the Databricks `deployment` job (Linux) are not affected. Prefer registering models meant for serving from Databricks.

## 3. A/B testing notebook (`notebooks/lecture6.ab_testing.py`)

| Problem | Error / effect | Fix |
|---|---|---|
| Model B's "different" parameters were identical to the config values used by model A | The A/B test compared a model with itself | Model B now uses `learning_rate=0.05, n_estimators=300, max_depth=3` |
| The inline A/B wrapper had the same Windows path bug as #2 | `OSError: No such file or directory: '/model/artifacts\...'` on the endpoint | `.replace("\\", "/")` on both artifact paths in `load_context`. The wrapper is defined in the notebook, so the fix is pickled with the model; no wheel rebuild needed |
| The endpoint was called with a token created at the top of the notebook with `lifetime_seconds=1200` | Training plus endpoint start-up took longer than 20 minutes: `403 {"error_code":403,"message":"Invalid access token."}` | `call_endpoint` uses `w.config.host` and `w.config.authenticate()`, which returns a fresh auth header on every call |
| `WorkspaceClient()` had no profile; the local `DEFAULT` profile is empty and `dev-databricks` points at another workspace | The client could resolve to the wrong workspace, or fail with `cannot configure default credentials` | `WorkspaceClient(profile="dbc-07d12638-ef42")`, the same workspace as the Spark session; the endpoint-creation cell reuses that client |

## 4. Environment names: `acc`/`prd` to `stage`/`prod`

- **Problem:** the bundle targets were renamed to `dev`/`stage`/`prod`, but the jobs pass `--env ${bundle.target}` to the scripts, and the config and its validation still used `acc`/`prd`.
- **Error:** every `stage`/`prod` job would fail on its first task with `ValueError: Invalid environment: prod. Expected 'prd', 'acc', or 'dev'`, and CD would fail on an unknown bundle target.
- **Fix:** renamed consistently in `databricks.yml` (targets and `stage_` name prefix), `project_config_marvel.yml` (`stage`/`prod` sections, catalogs `mlops_stage`/`mlops_prod`), `src/marvel_characters/config.py` (validation), `.github/workflows/cd.yml` (matrix `[stage, prod]`, tag on `prod`) and `CLAUDE.md`.
- **Still required outside the code:** Unity Catalog catalogs `mlops_stage` and `mlops_prod`, and GitHub environments named exactly `stage` and `prod`, each with the variable `DATABRICKS_HOST` and the secrets `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` (service principal OAuth).

## 5. CI/CD release tag never created

- **Problem:** `.github/workflows/cd.yml` and `ci.yml` used `echo "VERSION=$(cat version.txt)"`, which only prints text; it does not set a variable.
- **Error:** none, which is the problem: `$VERSION` was empty, so `git tag $VERSION` just listed tags and `git push origin $VERSION` pushed the branch. `prod` releases were silently never tagged.
- **Fix:** `VERSION=$(cat version.txt)` followed by `git tag "$VERSION"` / `git push origin "$VERSION"`. Because the tag is now really created, **bump `version.txt` before each merge to `main`**. The CI tag step fails early on a pull request that reuses an existing version.

## 6. `preprocessing` job task fails writing `train_set` (INT vs BIGINT)

- **Problem:** `DataProcessor.preprocess()` used `.astype("int")` / `.astype(int)` for `Teams` and `Alive`. With numpy 1.x that is **int32 on Windows** but **int64 on Linux**. The tables were first written from a laptop (lecture 2 via Databricks Connect), so Delta stored those columns as `INT`; the `deployment` job then ran `process_data.py` on Linux and produced `BIGINT`. `mode("overwrite")` replaces the data but keeps the table schema, and Delta refuses to change the type implicitly.
- **Error:** `databricks bundle run deployment` failed at the `preprocessing` task, skipping all downstream tasks:
  `[DELTA_FAILED_TO_MERGE_FIELDS] Failed to merge fields 'Teams' and 'Teams'` caused by `[DELTA_MERGE_INCOMPATIBLE_DATATYPE] Failed to merge incompatible data types IntegerType and LongType`.
- **Fix:** in `src/marvel_characters/data_processor.py`:
  - `Teams` and `Alive` use `.astype("int64")`, so the schema is the same whichever OS writes the tables.
  - `save_to_catalog` writes with `.option("overwriteSchema", "true")`, so a full overwrite also replaces column types left by earlier writes (needed once to move the existing `INT` columns to `BIGINT`).
- Redeploy (`databricks bundle deploy`) before re-running the job; the job uses the wheel uploaded at deploy time.

## 7. CI/CD setup: service principals, GitHub environments, stage → prod order

- **Problem:** CD had never run: no GitHub environments, no secrets, no service principals. Also `cd.yml` deployed with a matrix `[stage, prod]`, which starts **both deployments in parallel**. The prod approval gate could be approved before stage finished, or even after stage failed.
- **Error / risk:** CD would fail at authentication; once credentials existed, untested changes could reach prod before stage had deployed successfully.
- **Fix:**
  - **Databricks:** one service principal per environment, `stage_spn` and `prod_spn`, each with an OAuth secret. Each can use only its own catalog: `USE_CATALOG` on `mlops_stage` / `mlops_prod`, and `USE_SCHEMA, CREATE_TABLE, CREATE_MODEL, SELECT, MODIFY, EXECUTE` on `<catalog>.marvel_characters`.
  - **GitHub → Settings → Environments:** `stage` and `prod`, each with secrets `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` (that environment's service principal) and variable `DATABRICKS_HOST`. `prod` has **required reviewers**. There is no `dev` environment: no workflow uses one, because dev is deployed from a laptop with a personal login.
  - **`.github/workflows/cd.yml`:** two jobs instead of a matrix. `deploy-stage` runs first; `deploy-prod` has `needs: deploy-stage`, so it is skipped if stage fails and then waits for approval. Only the prod job gets `contents: write`, for the release tag. A `concurrency` group stops two CD runs deploying over each other. The unused step that wrote the secret into `~/.databrickscfg` was removed; the CLI authenticates from the `DATABRICKS_*` environment variables.
- **Flow:** feature branch → pull request (CI: lint + tests) → merge to `main` → CD deploys stage → approve → CD deploys prod and tags `version.txt`.
- **Not yet covered:** when the stage/prod `deployment` jobs run and reach `deploy_model`, the service principal needs `CAN_MANAGE` on the endpoint `marvel-character-model-serving`, which `scripts/deploy_model.py` uses for every environment.

## Local environment notes (Windows)

- **Do not install the `test` extra into `.venv`.** `pyspark` (test) and `databricks-connect` (dev) both provide the `pyspark` module and break each other, and `uv sync --extra test` removes the dev extra. Run tests in a throwaway environment:
  ```
  uv run --isolated --extra test pytest -m "not ci_exclude"
  ```
- **No `[dependency-groups] dev` in `pyproject.toml`.** The VS Code Databricks extension added `[dependency-groups] dev = ["databricks-connect~=16.4.0"]`. uv installs the `dev` group on every `uv sync` by default, including CI's `uv sync --extra test`, so `databricks-connect` and `pyspark` were installed together and test collection failed with `ImportError: cannot import name '_with_origin' from 'pyspark.errors.utils'`. The group was removed; `databricks-connect` comes from the `dev` extra. An older local uv (0.5.x) also rewrites `uv.lock` in an older format (drops `upload-time`), so do not commit a relock that only changes formatting.
- **Close notebooks before `uv sync`.** A running Jupyter kernel locks `debugpy` files; the sync then stops halfway with `Access is denied (os error 5)` and can leave `ipykernel` uninstalled. Close the kernel and re-run `uv sync --extra dev`.
- **Keep the Databricks CLI current, and installed only once.** An old Chocolatey install (v0.240.0) sat ahead of the winget install (v1.18.0) on `PATH`. `databricks bundle validate`/`deploy` download Terraform and verify it with a HashiCorp signing key built into the CLI; the key in v0.240.0 has expired, so they failed with `Error: error downloading Terraform: unable to verify checksums signature: openpgp: key expired`. Fixed by `choco uninstall databricks-cli` (admin PowerShell); check with `databricks --version`. CD had the same exposure: `.github/workflows/cd.yml` pinned `databricks/setup-cli` and the CLI to v0.246.0, so both were bumped to v1.18.0 (action pinned by commit SHA `6a2e75f`).
- **Use a PowerShell terminal in VS Code.** The Databricks extension's "Run as file with Databricks Connect" sends PowerShell syntax (`& "...python.exe" ...`); in Git Bash it fails with `syntax error near unexpected token '&'`.
