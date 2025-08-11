
LABELS_INT_TO_STR = {
    -1 : "static obstacle",
      0: "unknown",
      1: "free",
      2: "clean",
      3: "base"
}

LABELS_STR_TO_INT = {
    "static obstacle": -1,
    "unknown" : 0,
    "free": 1,
    "clean": 2,
    "base": 3
}

# -2 means "out of map"

ACTION_TO_STRING = ['right', 'down', 'left', 'up']
