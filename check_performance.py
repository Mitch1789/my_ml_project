import json
import sys

# 1. Load metrics
with open("artifacts/metrics.json") as f:
    metrics = json.load(f)

acc = metrics.get("accuracy", 0)
threshold = 0.80

# 2. Compare to baseline
if acc < threshold:
    print(f"❌ Accuracy {acc:.4f} below threshold {threshold}")
    sys.exit(1)
else:
    print(f"✅ Accuracy {acc:.4f} meets threshold {threshold}")
    sys.exit(0)
