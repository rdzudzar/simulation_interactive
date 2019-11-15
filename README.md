# simulation_interactive
Interactive exploration of the Dark Sage output obtained from the mini Millennium simulation

#To run it directly with the Dark Sage output from OzStar:

bokeh serve --allow-websocket-origin=localhost:3112 --port=3112 GUI_DarkSage.py

Then forward localhost:3112 to local computer, so on local machine:
ssh username@ozstar.swin.edu.au -L 3112:ozstar:3112

Copy-paste url link from bokeh serve to the local browser
