"""
🚀 Phase 3.3 — Model Registry (Recursive Detection Fix)
Finds model artifacts in nested folders and registers them.
"""

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Hospital_Readmission_Tracking"
MODEL_NAME = "Hospital_Readmission_Model"

print("🚀 Phase 3.3: Registering Models in MLflow Registry (Recursive Fix)")

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# ---------------------------
# Helper: recursive artifact search
# ---------------------------


def find_model_artifact(run_id, path=""):
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            result = find_model_artifact(run_id, item.path)
            if result:
                return result
        elif "MLmodel" in item.path or item.path.endswith("model.ubj"):
            # The folder that contains MLmodel is the model artifact path
            return "/".join(item.path.split("/")[:-1])
    return None


# ---------------------------
# Search and register models
# ---------------------------
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)

registered = 0

for run in runs:
    run_id = run.info.run_id
    acc = run.data.metrics.get("accuracy", 0)
    f1 = run.data.metrics.get("f1_score", 0)

    print(f"\n🔍 Checking run {run_id} | Acc={acc:.4f} | F1={f1:.4f}")

    model_path = find_model_artifact(run_id)
    if not model_path:
        print("⚠️  No model artifact found — skipping this run.")
        continue

    print(f"✅ Found model artifact at '{model_path}' — registering...")
    model_uri = f"runs:/{run_id}/{model_path}"

    try:
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, result.version, "accuracy", str(acc))
        client.set_model_version_tag(MODEL_NAME, result.version, "f1_score", str(f1))

        # Assign stages
        if registered == 0:
            client.transition_model_version_stage(MODEL_NAME, result.version, "Production")
        elif registered == 1:
            client.transition_model_version_stage(MODEL_NAME, result.version, "Staging")
        else:
            client.transition_model_version_stage(MODEL_NAME, result.version, "Archived")

        print(f"✅ Registered model version {result.version} ({MODEL_NAME}) successfully!")
        registered += 1

    except Exception as e:
        print(f"❌ Failed to register model: {e}")

print(f"\n🎯 Phase 3.3 complete — {registered} model(s) registered successfully!")
print("💾 Check 'Models' tab in MLflow UI to confirm.")
