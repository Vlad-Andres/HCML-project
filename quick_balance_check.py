from src import data_loader
from src.feature_engineering import build_base_table
from src.data_loader import resolve_data_dir, load_all_tables, inspect_tables


DATA_DIR = resolve_data_dir()
tables = load_all_tables(DATA_DIR)

inspect_tables(tables["student_info"], tables["courses"])
# base = build_base_table(tables["student_info"], tables["courses"])
# print(len(base[base["final_result"] == 0]))
# print(len(base[base["final_result"] == 1]))
