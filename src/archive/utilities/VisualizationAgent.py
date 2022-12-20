"""
This script encapsulates the operations involved in pulling the performance metrics of our MAESTRO-X numerical
evaluations from the Plotly API (saved as *.xlsx files) and visualizing them in matplotlib for cleaner figures.

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2022. All Rights Reserved.
"""

import numpy as np
import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def visualize(file_name: str, trace_names: List, plot_type: str, x_limits: Tuple, y_limits: Tuple,
              x_label: str, y_label: str, x_font: Dict, y_font: Dict, legend_loc: str, title: str, title_font: str,
              marker_format: Dict, line_format: Dict, format_string: str, image_format: str, resolution: int) -> int:
    """
    Visualize the metrics in the <filename,tracename/sheetname> pair in matplotlib
    """

    image_name = ''.join([file_name.rsplit('.')[0], '.', image_format])

    try:

        for trace_name in trace_names:
            df = pd.read_excel(file_name, sheet_name=trace_name, engine='openpyxl')
            x, y = np.array(df[x_label]), np.array(df[y_label])

            if plot_type == 'LINES+MARKERS':
                plt.plot(x, y, fmt_str=format_string, label=trace_name)
            elif plot_type == 'SCATTER':
                plt.plot(x, y, marker=marker_format['style'], markersize=marker_format['size'], label=trace_name)
            elif plot_type == 'LINES':
                plt.plot(x, y, linestyle=line_format['style'], markersize=marker_format['size'], label=trace_name)
            else:
                raise NotImplementedError('The plot_type {} has not been implemented in this agent'.format(plot_type))

    except NotImplementedError as nie:
        print('[ERROR] MAESTRO-X VisualizationAgent | Feature not available: {}'.format(tb.print_tb(nie.__traceback__)))
        return 1

    plt.grid()
    plt.legend(loc=legend_loc)
    plt.xlim(x_limits[0], x_limits[1])
    plt.ylim(y_limits[0], y_limits[1])
    plt.title(title=title, fontdict=title_font)
    plt.xlabel(xlabel=x_label, fontdict=x_font)
    plt.ylabel(ylabel=y_label, fontdict=y_font)
    plt.savefig(image_name, bbox_inches='tight', format=image_format, dpi=resolution)

    print('[INFO] MAESTRO-X VisualizationAgent | The data from {} has been visualized in '
          'matplotlib as a {} plot and is available at {}'.format(file_name, plot_type, image_name))
    return 0
