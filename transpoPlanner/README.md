# communitySim

### Generate Transportation Map from Coordinates
To generate a .osm file storing the transportation graph of a location, use generateGraph.py! \\
Example: \\
For generating a graph of Duke's East Campus, I got the coordinates of Lily library from Google Maps. \\
Then, I use generateGraph.py with args --lat, --lon, --radius, --tolerance, --building_yaml, --highway_yaml, --osm_filename. \\
python generateGraph.py 36.00780585133558 -78.91533560763814 500 5 building.yaml highway.yaml lily.osm \\
Or, use the interactive python notebook generateGraph.ipynb. \\