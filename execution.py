from testing import run_experiments
from visualizations import generate_visualizations

results_df, y = run_experiments()

generate_visualizations(results_df, y)