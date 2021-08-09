import pandas as pd

def get_dataframe_for_start_end_points():
    start_end_points = pd.DataFrame()
    start_end_points["run"] = []
    start_end_points["segment"] = []
    start_end_points["component"] = []
    start_end_points["start_or_end"] = []
    start_end_points["point_x"] = []
    start_end_points["point_y"] = []
    start_end_points["point_z"] = []

    return start_end_points