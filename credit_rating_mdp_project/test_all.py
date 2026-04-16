"""Quick test: generate data, train model, predict, run MRP/MDP."""
import sys
sys.path.insert(0, '.')

from src.data_preprocessing import initialise_data
initialise_data()
print("✅ Data generated")

from src.train_rating_model import train_and_select
name, model, metrics, *_ = train_and_select()
print(f"✅ Best model: {name}")
print(f"   Test accuracy: {metrics['test_accuracy']:.3f}")
print(f"   Test F1-macro: {metrics['test_f1_macro']:.3f}")

from src.predict_rating import predict_single
from src.data_preprocessing import lookup_company, load_dataset
df = load_dataset()
row = lookup_company("Tata Motors", df)
if row is not None:
    res = predict_single(row.to_dict())
    print(f"✅ Tata Motors predicted: {res['predicted_rating']} "
          f"(confidence {res['confidence']*100:.1f}%)")

from src.data_preprocessing import load_transition_matrix
from src.mrp_model import MRP
tm = load_transition_matrix()
mrp = MRP(tm, gamma=0.95)
mrp.compute_state_values()
print(f"✅ MRP state values: {dict(zip(mrp.states, mrp.V.round(2)))}")

from src.mdp_model import MDP
mdp = MDP(tm.values, gamma=0.95)
V, policy, hist = mdp.value_iteration()
pt = mdp.get_policy_table(policy)
print(f"✅ MDP converged in {len(hist)} iterations")
print(f"✅ Policy table:\n{pt}")

print("\n🎉 All modules working correctly!")
