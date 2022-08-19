import numpy as np
from scipy import signal
import cv2
import time
import SeamCarving

Start      = time.time()

print('Processing...')
New_Width  = 200 # Number of Columns
New_Height = 358 # Number of Rows
Image      = cv2.imread('test.png')

Output = SeamCarving.SeamCarver(Image, New_Width, New_Height)
cv2.destroyAllWindows()
cv2.imwrite('Bruh.png', Output)


End = time.time()           
print(End - Start)




