System Architecture Diagram
====================================================

Legend:
  ->  : Direct import (A imports B)
  <-> : Mutual import (rare, but shown if present)
  [*] : Uses external library (not shown in arrows)

Modules:
---------
main.py
    |
    |---> ui_functions.py
    |---> general_functions.py
    |---> yolo_functions.py
    |---> ball_functions.py
    |---> ruck_functions.py
    |---> lineout_functions.py
    |---> offside_functions.py
    |---> field_functions.py
    |---> point_functions.py
    |---> drawing_functions.py
    |---> constants.py

ui_functions.py
    |---> constants.py
    |---> tkinter [*]
    |---> cv2 [*]
    |---> numpy [*]

general_functions.py
    |---> cv2 [*]
    |---> numpy [*]

yolo_functions.py
    |---> ultralytics.YOLO [*]
    |---> cv2 [*]
    |---> general_functions.py

ball_functions.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> general_functions.py
    |---> drawing_functions.py
    |---> constants.py

field_functions.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> sklearn.cluster.KMeans, DBSCAN [*]
    |---> matplotlib [*]
    |---> general_functions.py
    |---> line_functions.py

line_functions.py
    |---> numpy [*]
    |---> scipy.stats [*]

point_functions.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> matplotlib [*]
    |---> field_functions.py
    |---> general_functions.py
    |---> drawing_functions.py
    |---> line_functions.py
    |---> constants.py

drawing_functions.py
    |---> cv2 [*]
    |---> matplotlib.pyplot [*]

offside_functions.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> sklearn.cluster.KMeans [*]
    |---> general_functions.py

ruck_functions.py
    |---> constants.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> general_functions.py
    |---> drawing_functions.py
    |---> ball_functions.py

lineout_functions.py
    |---> constants.py
    |---> cv2 [*]
    |---> numpy [*]
    |---> general_functions.py
    |---> drawing_functions.py
    |---> ball_functions.py

constants.py
    (no imports)
