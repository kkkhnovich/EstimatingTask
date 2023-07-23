import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import Image, display


# class for displaying plots and coefficients of determination
class ModelEvaluator:
    # class for drawing and saving plots
    class PlotDrawer:
        def __init__(self):
            self.folder_name = "plots"
            if not os.path.exists(self.folder_name):
                os.makedirs(self.folder_name)
            self.plot_paths = []

        def draw_plots(self, json_file):
            # Residual plots to show homoscedasticity.
            dataset = pd.read_json(json_file)
            for i in ['floor', 'ceiling']:
                for j in ['mean', 'max', 'min']:
                    sns.scatterplot(x=dataset.index, y=dataset[f"{i}_{j}"])
                    plot_path = os.path.join(self.folder_name, f"Residual_Plot_{i}_{j}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    self.plot_paths.append(os.path.abspath(plot_path))
            return self.plot_paths

    def __init__(self):
        self.plot_drawer = self.PlotDrawer()
        self.r_2 = {}

    # This function calculate coefficients of determination for floor and ceiling.
    # It saves them to a dictionary.
    def _calculate_statistics(self, json_file):
        def modify_row(row):
            for j in range(1, row['gt_corners'] + 1):
                row[f"{part}{j}"] = row['angle']
            return row

        dataset = pd.read_json(json_file)
        for part in ['floor', 'ceiling']:
            angles = dataset.drop(
                ['mean', 'max', 'min', 'rb_corners', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean',
                 'ceiling_max',
                 'ceiling_min'], axis=1)
            angles['angle'] = (angles['gt_corners'] - 2) * 180 / angles['gt_corners']
            for i in range(1, 11):
                angles[f"{part}{i}"] = 0

            angles = angles.apply(modify_row, axis=1)
            mean_values = angles.drop(['name', 'gt_corners', 'angle'], axis=1).mean()
            residuals = dataset.drop(
                ['mean', 'max', 'min', 'name', 'floor_max', 'floor_min', 'gt_corners', 'ceiling_max', 'ceiling_min',
                 'rb_corners'],
                axis=1)
            rss = 4 * residuals[f"{part}_mean"].sum()
            from_mean_deviations = pd.DataFrame()
            for i in range(1, 10):
                from_mean_deviations[f"{part}{i}"] = (angles[f"{part}{i}"] - mean_values[f"{part}{i}"]) ** 2
            tss = 0
            for i in range(1, 10):
                tss += from_mean_deviations[f"{part}{i}"].sum()
            self.r_2[part] = 1 - rss / tss

    # This function displays coefficients of determination.
    def display_statistics(self, json_file):
        self._calculate_statistics(json_file)
        for i in self.r_2:
            print(f"Coefficient of determination for {i}'s angles:{self.r_2[i]}")

    # This function displays residual plots.
    def display_plots(self, json_file):
        self.plot_drawer.draw_plots(json_file)
        for path in self.plot_drawer.plot_paths:
            display(Image(filename=path))
