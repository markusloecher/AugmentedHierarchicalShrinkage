"""
    Gain difference plots

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-02-14
"""

import matplotlib.pyplot as plt


def gain_diff_plots(node_list: list):
    """
    Create the gain difference plots, before and after permutation.
    Plot the gain difference in relation with the number of rows -
    number of elements that are permuted

    :param node_list: Nodes list
    """

    # [1.] Nodes list --------------------------------------------------------------------------------------------------
    x_list = []
    y_list = []

    for node in node_list:

        if node.gain_before_permutation is not None and \
                node.gain_after_permutation is not None and \
                node.permuted_rows_nr is not None:

            x_list.append(node.permuted_rows_nr)
            y_list.append(node.gain_after_permutation - node.gain_before_permutation)

    # [2.] Scatter plot ------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))

    plt.scatter(x_list, y_list, s=8, alpha=0.5)
    plt.xlabel("Number of rows of permuted feature", fontsize=12, fontweight='bold')
    plt.ylabel("Gain difference", fontsize=12, fontweight='bold')
    plt.title("Gain difference w.r.t. permuted feature rows", fontsize=16, fontweight='bold')
    plt.show()
    # fig.savefig(os.path.join(output_data_path, f"Value_Range_{feature_name}.png"))
    plt.close()

