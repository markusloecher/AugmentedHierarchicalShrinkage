"""
    Feature importance plots

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-02-16
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances_hist_matplotlib(features_column_names: list,
                                             feature_importances: list):
    """
    Plot the feature importances in a horizontal histogram with matplotlib

    :param features_column_names: Features column names
    :param feature_importances: The corresponding feature importances
    :return:
    """

    # output_data_path = os.path.join(os.path.join("data", "output_data", "plots", "explanations"))

    fig, ax = plt.subplots(figsize=(12, 12))

    y_pos = np.arange(len(features_column_names))

    ax.barh(y_pos, feature_importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_column_names, fontsize=24, fontweight='bold')
    ax.invert_yaxis()
    ax.set_title('Feature Importances', fontsize=32, fontweight='bold')
    plt.tight_layout()
    plt.show()
    # fig.savefig(os.path.join(output_data_path, f"explanations_feature_importances.png"))
    plt.close()
