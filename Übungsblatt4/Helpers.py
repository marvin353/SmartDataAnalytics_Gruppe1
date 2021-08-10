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

def get_relative_start_end_points(df, start_end_points):
    # clean start_end_points
    start_end_points = start_end_points.iloc[0:0]

    # Loop through all runs
    for run in df['run'].unique():

        df_run = df[df["run"] == run]

        # Iterate over Components (1 component has 3 segments)
        for i in range(int(len(df_run["segment"].unique()) / 3)):
            # print("i: " + str(i))
            for index, seg in enumerate(df_run["segment"].unique()[i * 3:(i + 1) * 3]):  # [0:i*3]):
                if not index % 3 == 2:
                    continue
                # print("-----")
                # print(str(index) + ", " + str(index%3) + ", Seg: " + str(seg))
                t = df_run[df_run["segment"] == seg]

                # We set the startpoint as 0,0,0 for x,y and z direction
                new_row_start = {'run': run, 'segment': seg, 'component': i + 1, 'start_or_end': 'start', 'point_x': 0,
                                 'point_y': 0, 'point_z': 0}
                # We now calculate the relative endpoint value by subtracting the startponit form the endpoint separately for x,y and z direction.
                new_row_end = {'run': run, 'segment': seg, 'component': i + 1, 'start_or_end': 'end',
                               'point_x': t["pos_x"].iloc[-1] - t["pos_x"].iloc[0],
                               'point_y': t["pos_y"].iloc[-1] - t["pos_y"].iloc[0],
                               'point_z': t["pos_z"].iloc[-1] - t["pos_z"].iloc[0]}

                start_end_points = start_end_points.append(new_row_start, ignore_index=True)
                start_end_points = start_end_points.append(new_row_end, ignore_index=True)

    return start_end_points