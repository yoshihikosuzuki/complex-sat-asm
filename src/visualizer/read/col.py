import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex, XKCD_COLORS

# up to ~1000 colors
ID_TO_COL = {i: col for i, col in enumerate(XKCD_COLORS.values())}
DIST_TO_COL = np.vectorize(lambda x: rgb2hex(cm.Blues_r(x)))
