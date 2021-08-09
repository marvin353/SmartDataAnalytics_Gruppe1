# Importe

## Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns

## Graph modules
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from mpl_toolkits.mplot3d import Axes3D

import os.path
from enum import Enum

colors = {0:'tab:blue',1:'tab:orange',2:'tab:green'}

def get_search_segments(data):
    runs = data["run"].unique()
    split_data = []

    # Da für die Bearbeitung nur die Suchen interessant sind, hier schon eine Filterung
    search_list = []

    for r in runs:
        df = data[data["run"]== r]
        segments = df["segment"].unique()
        count_segments = len(segments)
        count_components = int(count_segments/3)
        # print(f"{r} : {count_segments = } {count_components = } \n {segments}")
        run_data = []
        for i in range(count_components):
            move = df[df["segment"] == segments[i*3]]
            descend = df[df["segment"] == segments[i*3 + 1]]
            search = df[df["segment"] == segments[i*3 + 2]]
            component_data = [move, descend, search]
            run_data.append(component_data)
            search_list.append(search)
        split_data.append(run_data)
    return search_list, split_data

def get_start_end_points(data):
    search_list, _ = get_search_segments(data)

    start_points = []
    end_points = []
    rel_end_points = []

    for r in search_list:
        x = r["pos_x"]
        y = r["pos_y"]
        z = r["pos_z"]

        point = r[["pos_x", "pos_y", "pos_z"]]

        start_idx = r["timestamp"].idxmin()
        end_idx = r["timestamp"].idxmax()

        start = point.loc[start_idx].values
        end = point.loc[end_idx].values
        # As it is more interessting to see the end point in relation to the start point the coordinates from the start get substracted.
        rel_end = end - start

        #Adding points to the lists
        start_points.append(start)
        end_points.append(end)
        rel_end_points.append(rel_end)

        dist = point.diff().fillna(0)

        # Order ist important here! For the first calculation we therefore can use the whole DataFrame without selectors.
        dist["distance"] = np.linalg.norm(dist.values,axis=1)
        dist["distance2d"] = np.linalg.norm(dist[["pos_x", "pos_y"]].values, axis=1)

    start_points = pd.DataFrame(start_points, columns=["x","y","z"])
    end_points = pd.DataFrame(end_points, columns=["x","y","z"])
    rel_end_points = pd.DataFrame(rel_end_points, columns=["x","y","z"])
    return start_points, end_points, rel_end_points

def plot_search2d(search_list):
    for r in search_list:
        segment = r["segment"].iloc[0]
        x = r["pos_x"]
        y = r["pos_y"]

        plt.plot(x,y)
        plt.title(f"{segment =  }")
        plt.show()

def plot_data(start_points, end_points, rel_end_points):
    sns.set(rc={'figure.figsize': (15,10)})
    # Prepare data for seaborn scatterplot
    rel_plot_data = rel_end_points.copy()
    rel_plot_data["kind"] = "end"
    index = rel_plot_data.columns
    rel_plot_data = rel_plot_data.append(pd.DataFrame([[0,0,0,"start"]], columns=index))

    # print("Start points")
    sns.scatterplot(data=start_points, x="x", y="y")
    plt.title("Start points")
    plt.show()

    #print("\nEnd points")
    sns.scatterplot(data=end_points, x="x", y="y")
    plt.title("End points")
    plt.show()

    print()
    #print("\nRelative end points")
    sns.scatterplot(data=rel_plot_data, x="x", y="y", hue="kind")
    plt.title("Relative end points")
    plt.show()

    sns.displot(data=rel_end_points, x="x", y="y")
    plt.title("Distribution plot relative end points")
    plt.show()

    sns.jointplot(data=rel_plot_data, x="x", y="y", hue="kind", )
    plt.title("Joint plot relative end points")
    plt.show()

def mean_between_neighbors(array):
    result =  0.5*(array[1:] + array[:-1])
    return ['{:,.3f}'.format(v) for v in result]

def generate_histogram2d(rel_end_points, x_name = "x", y_name = "y", step_size: float = 0.02):
    x = rel_end_points[x_name]
    y = rel_end_points[y_name]

    x_min = rel_end_points[x_name].min()
    x_max = rel_end_points[x_name].max()
    y_min = rel_end_points[y_name].min()
    y_max = rel_end_points[y_name].max()

    x_ll = ((x_min//step_size) * step_size) - (0.5 * step_size)
    x_ul = ((x_max//step_size + 2) * step_size) + (0.5 * step_size)

    y_ll = ((y_min//step_size) * step_size) - (0.5 * step_size)
    y_ul = ((y_max//step_size + 2) * step_size) + (0.5 * step_size)

    x_edge = np.arange(x_ll, x_ul, step_size)
    y_edge = np.arange(y_ll, y_ul, step_size)

    hist, xedge, yedge = np.histogram2d(x=x, y=y, bins=(x_edge,y_edge))
    filepath = os.path.join("files","hist2d_out.csv")
    np.savetxt(filepath,hist,delimiter=";",fmt="%d")

    return hist, xedge, yedge

class direction(Enum):
    # directions
    S = (0, -1)
    N = (0, 1)
    W = (-1, 0)
    E = (1, 0)

class turn(Enum):
    turn_left, turn_right = {direction.N.value: direction.E.value, direction.E.value: direction.S.value, direction.S.value: direction.W.value, direction.W.value: direction.N.value}, {direction.N.value: direction.W.value, direction.W.value: direction.S.value, direction.S.value: direction.E.value, direction.E.value: direction.N.value} # old -> new direction

def build_spiral(size_x: int, size_y: int, start_x: int, start_y: int, start_direction: direction, rotation: turn):
    assert(size_x > 0)
    assert(size_y > 0)
    assert(0 <= start_x < size_x)
    assert(0 <= start_y < size_y)
    assert(isinstance(start_direction, direction))
    assert(isinstance(rotation, turn))

    width = max(start_x, size_x - (start_x + 1)) * 2 + 1
    height = max(start_y, size_y - (start_y + 1)) * 2 + 1

    # only with squared arrays will there be no edge cases
    length = max(width, height)

    x, y = length // 2, length // 2 # start in the center
    dx, dy = start_direction.value
    matrix = [[None] * length for _ in range(length)]
    #print(matrix)
    count = 0

    while True:
        matrix[y][x] = count
        count += 1
        #print(matrix)

        new_dx, new_dy = rotation.value[dx,dy]
        new_x, new_y = x + new_dx, y + new_dy

        if (0 <= new_x < length and 0 <= new_y < length and matrix[new_y][new_x] is None): # can turn right
            x, y = new_x, new_y
            dx, dy = new_dx, new_dy
        else: # try to move straight
            x, y = x + dx, y + dy
            if not (0 <= x < length and 0 <= y < length):
                result = np.array(matrix)
                #print(result)
                # lower bounds for the slice to match the wanted dimensions
                x_lb = length//2 - start_x
                #print(x_lb)
                y_lb = length//2 - start_y
                #print(y_lb)
                return result[x_lb: (x_lb + size_x), y_lb:(y_lb + size_y)]