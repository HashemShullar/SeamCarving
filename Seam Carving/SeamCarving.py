import numpy as np
from scipy import signal
import cv2
import time


def ImEnergy(Image):
    kernely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernelx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

    b, g, r = cv2.split(Image)

    b_gradx   = signal.convolve2d(b, kernelx, mode='same')
    b_grady   = signal.convolve2d(b, kernely, mode='same')
    b_energy  = abs(b_gradx) + abs(b_grady)

    g_gradx   = signal.convolve2d(g, kernelx, mode='same')
    g_grady   = signal.convolve2d(g, kernely, mode='same')
    g_energy  = abs(g_gradx) + abs(g_grady)                             

    r_gradx   = signal.convolve2d(r, kernelx, mode='same')
    r_grady   = signal.convolve2d(r, kernely, mode='same')
    r_energy  = abs(r_gradx) + abs(r_grady)                             

    return b_energy + g_energy + r_energy


def CumulativeEnergy(Energy):
    M = np.zeros((Energy.shape[0], Energy.shape[1]))
    M[0,:] = Energy[0,:]
    for i in range(1, Energy.shape[0]):
        for j in range(Energy.shape[1]):
            if j == 0:
                M[i, j] = Energy[i, j] + min(M[i - 1, j], M[i - 1, j+1])
            elif j == (Energy.shape[1] - 1):
                M[i, j] = Energy[i, j] + min(M[i - 1, j-1], M[i - 1, j])
            else:
                M[i, j] = Energy[i, j] + min(M[i - 1, j-1], M[i - 1, j], M[i - 1, j+1])
    return M

def SeamFinder(M, k, Seam):
    Holder   = M
    Min      = min(Holder[M.shape[0]-1, :])
    Index    = list(np.where(Holder[M.shape[0]-1, :] == Min))
    Seam[k][0] = [M.shape[0]-1, Index[0][0]]
    Holder[M.shape[0]-1, Index[0][0]] = 9000000
    for i in reversed(range(M.shape[0]-1)):

        Min = min(Holder[i-1, Index[0][0]-1], Holder[i-1, Index[0][0]], Holder[i-1, Index[0][0]+1])
        if Min == Holder[i-1, Index[0][0]-1]:
            Index = [[Index[0][0]-1]]
            Holder[i-1, Index[0][0]-1]  = 9000000
        elif Min == Holder[i-1, Index[0][0]]:
            Index = [[Index[0][0]]]
            Holder[i-1, Index[0][0]] = 9000000
        else:
            Index = [[Index[0][0]+1]]
            Holder[i-1, Index[0][0]+1] = 900000
        Seam[k][M.shape[0] - 1 - i] = [i, Index[0][0]]
#         ImCopy[i, Index[0][0]] = [0, 0, 255]
    return Seam

def SeamRemoval(Seam, img, k, flag):
    for pixel in Seam:
        for col in reversed(range(1, int(pixel[1]) + 1)):
            img[int(pixel[0])][col] = img[int(pixel[0])][col - 1]
        img[int(pixel[0])][0] = [0, 0, 255]
        if flag:
            return img#[:, 1:Image.shape[1]]
        else:
            return img#[:, 1:Image.shape[0]]

def SeamCarver(Image, New_Width, New_Height):
    c = Image.shape[1] - New_Width
    r = Image.shape[0] - New_Height
    if c > 0:
        flag = 1
        NumSeams   = c
        Seam       = np.zeros((NumSeams, Image.shape[0], 2))
        ImCopy     = Image.copy()
        Energy     = ImEnergy(Image)


        for k in range(NumSeams):
            M      = CumulativeEnergy(Energy) # Cumulative Minimum Energy Calculation
            Seam   = SeamFinder(M, k, Seam) # Seam Computing
            SeamViz(ImCopy, Seam[k,:,:])
            for pixel in Seam[k][:][:]:
                for col in reversed(range(1, int(pixel[1]) + 1)):
                    ImCopy[int(pixel[0])][col] = ImCopy[int(pixel[0])][col - 1]
            ImCopy   = ImCopy[:, 1:Image.shape[1]] 
            Energy   = ImEnergy(ImCopy)
 
            if k != (NumSeams - 1):
                Energy = ImEnergy(ImCopy)
        
    if r > 0:
        flag = 0
        if c == 0:
            ImCopy     = Image.copy()
        ImCopy = cv2.rotate(ImCopy, cv2.cv2.ROTATE_90_CLOCKWISE)
        NumSeams   = r
        Seam       = np.zeros((NumSeams, ImCopy.shape[0], 2))
        Energy     = ImEnergy(ImCopy)


        for k in range(NumSeams):
            M      = CumulativeEnergy(Energy) # Cumulative Minimum Energy Calculation
            Seam   = SeamFinder(M, k, Seam) # Seam Computing
            SeamViz(ImCopy, Seam[k,:,:])
            for pixel in Seam[k][:][:]:
                for col in reversed(range(1, int(pixel[1]) + 1)):
                    ImCopy[int(pixel[0])][col] = ImCopy[int(pixel[0])][col - 1]
            ImCopy   = ImCopy[:, 1:Image.shape[0]] 
            Energy   = ImEnergy(ImCopy)
            if k != (NumSeams - 1):
                Energy = ImEnergy(ImCopy)
    cv2.destroyAllWindows()
    return cv2.rotate(ImCopy, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

def SeamViz(Image, Seam):
    Copy = Image.copy()
    Copy[Seam[:,0].astype(int), Seam[:,1].astype(int)] = [0, 0, 255]
    cv2.imshow('Seam', Copy)
    cv2.waitKey(1)   
    time.sleep(0.1)
#    cv2.destroyAllWindows() 
    
