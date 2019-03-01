import numpy as np
import pandas as pd

data = pd.DataFrame([1,5,7,2,4])

for i in range(5):
    randindex = data.reindex(np.random.permutation(data.index))
    newdata = pd.DataFrame(randindex.values)
    print(newdata)
