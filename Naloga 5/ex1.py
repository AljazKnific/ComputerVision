from UZ_utils import *
from a5_utils import *

def disparity(p_z):
    sez = []
    f = 0.0025
    T = 0.12
    
    for i in range(1, p_z + 1):
        sez.append((f * T) / (0.01 * i))
    """    
    plt.plot(np.array(sez))
    plt.show()
    """
    return 0

disparity(100)

#Razrpšenost je obratno sorazmerna z razdaljo.
