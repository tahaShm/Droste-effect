import numpy as np
import cv2
def generateLog(z, r1, r2): # logarithm mapping
    ans = 0
    value = abs(z)
    if  ((r1 <= value) and (value <= r2)):
        ans = np.log(z/r1)
    return ans
def generateRotation(z, r1, r2): #part 2 -------------------
    alpha = np.arctan(np.log(r2/r1) / (2*np.pi)) # to see the result of part two replace "generateLog" with  "generateRotation" at line 31, and "phase4" with "phase1" at line 82.
    f = np.cos(alpha)
    return z * f * np.power(np.e, 1j*alpha)
def generatePart2and4(z, f, alpha) : # rotation and exponential mapping
    w = z * f * np.power(np.e, 1j*alpha) # part 2 -----------------
    w = np.power(np.e, w)
    return w

# part 1 ----------------------------------
img = cv2.imread('clock.jpg',0)
rows = np.linspace(-1, 1, num = len(img))
cols = np.linspace(-1, 1, num = len(img[0]))
[X, Y] = np.meshgrid(rows, cols)
r = len(X)
c = len(Y)
z = X + 1j * Y
w = []
r2 = 0.9
r1 = 0.2
for i in range(r):
    for j in range(c):
        w.append(generateLog(z[i][j], r1, r2))
wx = np.real(w)
wy = np.imag(w)
wxMax = max(abs(wx))
wyMax = max(abs(wy))
xNew = ((wx/wxMax + 1)*(c/2))
yNew = ((wy/wyMax + 1)*(r/2))

xNew = xNew.reshape(img.shape)
yNew = yNew.reshape(img.shape)
phase1 = np.zeros([r, c, 3], dtype="uint8")

for i in range(r):
    for j in range(c):
        phase1[int(yNew[i][j]) - 1, int(xNew[i][j]) - 1] = img[i][j]


# part 3 ---------------------------------------

copies = 3 # number of phase1 repeats
xNew = np.tile(xNew, (copies,1))
yNew = np.tile(yNew, (copies,1))
for i in range(copies*r):
        yNew[i] = yNew[i] + int(i / r) * r
phase3 = np.zeros([r*copies, c, 3], dtype="uint8")

for i in range(copies*r):
    for j in range(c):
        phase3[int(yNew[i][j]) - 1, int(xNew[i][j]) - 1] = img[i % r][j]

# part 4 ---------------------------------------

alpha = np.arctan(np.log(r2/r1) / (2*np.pi))
f = np.cos(alpha)

wx3 = (xNew*2/r - 1)*wxMax # convert xNew of phase3 to wx
wy3 = (yNew*2/c - 1)*wyMax # convert yNew of phase3 to wy
z4 = wx3 + 1j * wy3
z4 = generatePart2and4(z4, f, alpha)
wx4 = np.real(z4)
wy4 = np.imag(z4) # new wx and wy
wxMax4 = np.max(np.abs(wx4))
wyMax4 = np.max(np.abs(wy4))
xNew4 = (wx4/wxMax4 + 1)*(r/2)
yNew4 = (wy4/wyMax4 + 1)*(c/2)
phase4 = np.zeros([r, c, 3], dtype = "uint8")

for i in range(copies*r):
    for j in range(c):
        phase4[int((yNew4[i][j]) - 1) , int(xNew4[i][j]) - 1] = img[i%r][j]

cv2.imshow('image', phase4)
cv2.waitKey(0)
cv2.destroyAllWindows()