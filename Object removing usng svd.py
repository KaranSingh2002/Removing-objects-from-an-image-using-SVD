import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import svd
from sympy import Matrix, init_printing
from os import listdir, getcwd
from os.path import isfile, join
from random import randint
from PIL import Image
from sklearn.decomposition import TruncatedSVD

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

init_printing()

def get_data_Matrix(mypath="peds"):
    cwd = getcwd()

    mypath = join(cwd, mypath)
    files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith(".") == False]

    img = Image.open(files[0])
    I = np.array(img)

    Length, Width = I.shape

    X = np.zeros((len(files), Length * Width))
    for i, file in enumerate(files):
        img = Image.open(file)
        I = np.array(img)
        X[i, :] = I.reshape(1, -1)
    return X, Length, Width

def read_image(image_path):
    img = Image.open(image_path)
    I = np.array(img)
    X = I.reshape(-1, I.shape[0] * I.shape[1])
    return X

image_path = r"C:\Users\karan singh\Downloads\Featured-image_Raahgiri-Day-Street-stories.jpg"

X_custom, Length_custom, Width_custom = read_image(image_path)

X = np.array([[1.0, 2], [2, 1], [3, 3]])

Matrix(X)

U, s, VT = svd(X, full_matrices=False)

Matrix(U)

S = np.diag(s)

Matrix(S)

Matrix(VT)

X_ = U @ S @ VT
X_ = np.round(X_)

Matrix(X_)

X_2 = s[0] * U[:, 0:1] @ VT[0:1, :] + s[1] * U[:, 1:2] @ VT[1:2, :]

Matrix(X_2)

X = np.array([[1, 2], [2, 4], [4, 8.0001]])

Matrix(X)

U, s, VT = svd(X, full_matrices=False)
S = np.diag(s)

Matrix(S)

X_hat = np.round(s[0] * U[:, 0:1] @ VT[0:1, :])

Matrix(X_hat)

L = 1
Xhat = U[:, :L] @ S[0:L, 0:L] @ VT[:L, :]

Matrix(Xhat)

print(f"With {L} singular value and its corresponding singular vectors, {s[0:L] / s.sum()} variance of X is explained")

plt.figure()
plt.plot(np.cumsum(s) / s.sum())
plt.xlabel('L')
plt.title('Cumulative explained singular value')
plt.tight_layout()
plt.show()

svd_ = TruncatedSVD(n_components=1, random_state=42)

Z = svd_.fit_transform(X)

Z

Xhat = svd_.inverse_transform(Z)
Matrix(np.round(Xhat))

X, Length, Width = get_data_Matrix(mypath="peds")

X.shape

for i in range(5):
    frame = randint(0, X.shape[0] - 1)
    plt.imshow(X[frame, :].reshape(Length, Width), cmap="gray")
    plt.title("frame: " + str(frame))
    plt.show()

U, s, VT = svd(X, full_matrices=False)
S = np.diag(s)

L = 1
Xhat = U[:, :L] @ S[0:L, 0:L] @ VT[:L, :]

plt.imshow(Xhat[0, :].reshape(Length, Width), cmap="gray")
plt.title('Truncated SVD L=1')
plt.show()
