import scipy.io
import numpy as np
mat = scipy.io.loadmat('/home/godsom/Dataset/imdb/imdb_crop/imdb.mat')

# print(mat['imdb'][0][0][2])
print(np.mean(mat['imdb'][0][0][3][0]))
print(np.argwhere(np.isnan(mat['imdb'][0][0][3][0])).size)
print(mat['imdb'][0][0][3][0].size)

# nan_id = np.argwhere(np.isnan(mat['imdb'][0][0][3][0]))
# mat = mat['imdb'][:,:,:,,:,~nan_id]
a = (mat['imdb'][0][0][3][0])
notnan_id = np.argwhere(~np.isnan(a))
print(notnan_id)
a = a[notnan_id]
b = (mat['imdb'][0][0][2][0])
b = b[notnan_id]
print(a.size)
print(b.size)
print(np.mean(a))

#
# print(np.mean(mat['imdb'][0][0][3][0]))
# print(np.argwhere(np.isnan(mat['imdb'][0][0][3][0])).size)
# print(mat['imdb'][0][0][3][0].size)
