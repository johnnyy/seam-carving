import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)    

def forward_energy(im):

    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return energy

def backward_energy(im):

   # xgrad  = cv2.filter2D(im, cv2.CV_64F, np.array([1,0,-1]))
   # ygrad = cv2.filter2D(im, cv2.CV_64F, np.array([1,0,-1]).T)
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag



def get_minimum_seam(im):
    h, w = im.shape[:2]
    M = backward_energy(im)

   
    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()


    return np.array(seam_idx), boolmask


def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


def add_seam(im, seam_idx):

    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output



def seams_removal(im, num_remove, rot=False):
    for i in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im)
        im = remove_seam(im, boolmask)
    return im


def seams_insertion(im, num_add, rot=False):
    seams_record = []
    temp_im = im.copy()
    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im)
        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im,boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im




def seam_carve(im, dy, dx):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    output = im
    if dx < 0:
        output = seams_removal(output, -dx)

    elif dx > 0:
        output = seams_insertion(output, dx)

    if dy < 0:
        output = rotate_image(output, True)
        output= seams_removal(output, -dy, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output= seams_insertion(output, dy, rot=True)
        output = rotate_image(output, False)
        
    return output


#dy, dx = args["dy"], args["dx"]
#assert dy is not None and dx is not None
im = cv2.imread(sys.argv[1])
print("O formato da imagem é {}x{}".format(im.shape[0],im.shape[1]))
valor = input("Informe o novo shape VALORxVALOR:")
y,x = valor.split('x')
dy=int(y) - im.shape[0]
dx= int(x) - im.shape[1]
print(dy,'x',dx)
out_name = sys.argv[2]
output = seam_carve(im, dy, dx)#, mask, args["vis"])
cv2.imwrite(out_name, output)

