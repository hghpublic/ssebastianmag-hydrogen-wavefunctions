"""
File: interactive_plot/interactive_plot.py
Description: Script to run the interactive plot app.

How to use: Execute script and open http://127.0.0.1:8050/ in your browser.
If there is a port conflict:
1. Change the port in the function call bellow.
2. Execute script again
3. Try opening http://127.0.0.1:NEWPORT/
"""

from interactive_plot_util import create_dash_app

plot_app = create_dash_app()
plot_app.run(debug=True, port=8050)
