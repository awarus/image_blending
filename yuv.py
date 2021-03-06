import math
import numpy as np

from ctypes import *
conv = CDLL('./conv.so')

conv.convolve2d_fl.restype = c_double
conv.convolve2d_fl.argtypes = [
    c_int,
    c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')]

from skimage.io import imread, imsave, imshow
import skimage.transform as sk_t
import scipy.signal

from skimage import img_as_float32, img_as_ubyte

from matplotlib.pyplot import hist
from scipy.ndimage import gaussian_filter

def gauss(coef, x, y):
    return((math.exp((-x*x-y*y)/(2*coef*coef)))/(2*math.pi*coef*coef))

def create_kernel(coef) :
    k = round(coef*6+1)
    k_bound = k//2
    sum = 0.0
    ar = []
    ar = np.zeros((k, k), dtype=np.float32)

    for i in range(0, 2*k_bound+1, 1):
        for j in range(0, 2*k_bound+1, 1):
            x = (k_bound) - i
            y = (k_bound) - j
            ar[i, j] = gauss(coef, x, y)
            sum = sum + ar[i, j]
    
    for i in range(-k_bound, k_bound+1, 1):
        for j in range(-k_bound, k_bound+1, 1):
            ar[i, j] = ar[i, j]/sum

    return ar

def fast_convolve(part, kern):
    h = part.shape[0]
    w = part.shape[1]

    part = np.ascontiguousarray(part, dtype=np.float32)

    val = conv.convolve2d_fl(h, w, part, kern);
    val = np.float64(val)
    return val

def g_filter_fast(img1, ar):
    img = img1.copy()

    k = ar.shape[0]
    kd2 = k//2

    img = np.pad(img, kd2, mode = 'symmetric')
    
    height = img.shape[0]
    width = img.shape[1]

    out = np.zeros((height-kd2*2, width-kd2*2))

    y = -1
    for i in range(kd2+1, height-kd2, 1):
        y += 1
        x = -1
        for j in range(kd2+1, width-kd2+1, 1):
            x +=1
            start = (i-(kd2+1), j-(kd2+1))
            end = (i+kd2, j+kd2)
            out[y, x] = fast_convolve(img[start[0]:end[0], start[1]:end[1]], ar)
            
    return out
    

def gauss_pyramid(img, sigma, n_layers):
    if n_layers is 0:
        return 0

    ar = create_kernel(sigma)
    layer = []
    
    layer.append(g_filter_fast(img, ar))
    
    for i in range(1, n_layers):
        print(i)
        layer.append(g_filter_fast(layer[i-1], ar))
    
    return layer

def laplas_pyramid(img, sigma, n_layers):
    laplas = []
    
    gauss = gauss_pyramid(img, sigma, n_layers)
    temp = img - gauss[0]
    laplas.append(temp)
    
    for i in range(1, n_layers-1):
        temp = gauss[i-1] - gauss[i]
        laplas.append(temp)
        
    temp = gauss[n_layers-1]
    laplas.append(temp)
        
    return laplas, gauss

img_a = imread('examples/seasons/a.png')
img_b = imread('examples/seasons/b.png')
mask = imread('examples/seasons/mask.png')
mask = (mask > 128)

img_fa = img_as_float32(img_a)
img_fb = img_as_float32(img_b)
img_mask = img_as_float32(mask)
print(img_mask[0, 0])
r_a = img_fa[:, :, 0].copy()
g_a = img_fa[:, :, 1].copy()
b_a = img_fa[:, :, 2].copy()

r_b = img_fb[:, :, 0].copy()
g_b = img_fb[:, :, 1].copy()
b_b = img_fb[:, :, 2].copy()

r_mask = img_mask[:, :, 0].copy()
g_mask = img_mask[:, :, 1].copy()
b_mask = img_mask[:, :, 2].copy()

y_a = 0.2126*r_a + 0.7152*g_a + 0.0722*b_a
y_b = 0.2126*r_b + 0.7152*g_b + 0.0722*b_b
y_mask = 0.2126*r_mask + 0.7152*g_mask + 0.0722*b_mask

u_a = -0.0999*r_a - 0.3360*g_a + 0.4360*b_a
u_b = -0.0999*r_b - 0.3360*g_b + 0.4360*b_b
u_mask = -0.0999*r_mask - 0.3360*g_mask + 0.4360*b_mask

v_a = 0.6150*r_a - 0.5586*g_a - 0.0563*b_a
v_b = 0.6150*r_b - 0.5586*g_b - 0.0563*b_b
v_mask = 0.6150*r_mask - 0.5586*g_mask - 0.0563*b_mask

sigma = 2
layers = 7

# y channel
la, ga = laplas_pyramid(y_a, sigma, layers)
lb, gb = laplas_pyramid(y_b, sigma, layers)
gm = gauss_pyramid(y_mask, sigma, layers)

#print(type(ga[0]), type(gb[0]), type(gm[0]))

ly = [] #result pyramid

for i in range(0, layers):
    temp = gm[i]*la[i]+(1-gm[i])*lb[i]
    ly.append(temp)

y_img = ly[0]
imsave('examples/seasons/out' + str(1) + '.png', ly[i])

for i in range(1, layers):
    imsave('examples/seasons/out' + str(i+1) + '.png', ly[i])
    y_img += ly[i]

imsave('examples/seasons/y.png', y_img)

# v channel
la, ga = laplas_pyramid(v_a, sigma, layers)
lb, gb = laplas_pyramid(v_b, sigma, layers)

lv = [] #result pyramid

for i in range(0, layers):
    temp = gm[i]*la[i]+(1-gm[i])*lb[i]
    lv.append(temp)

v_img = lv[0]

for i in range(1, layers):
    v_img += lv[i]

imsave('examples/seasons/v.png', v_img)

# u channel
la, ga = laplas_pyramid(u_a, sigma, layers)
lb, gb = laplas_pyramid(u_b, sigma, layers)

lu = [] #result pyramid

for i in range(0, layers):
    temp = gm[i]*la[i]+(1-gm[i])*lb[i]
    lu.append(temp)

u_img = lu[0]

for i in range(1, layers):
    u_img += lu[i]

imsave('examples/seasons/u.png', u_img)

r_out = y_img + 1.2803*v_img
g_out = y_img - 0.2148*u_img - 0.3805*v_img
b_out = y_img + 2.1279*u_img

out_img = np.zeros((r_out.shape[0], r_out.shape[1], 3), dtype=np.float32)
out_img[:, :, 0] = r_out
out_img[:, :, 1] = g_out
out_img[:, :, 2] = b_out

out_img = np.clip(out_img, 0, 1)

img2 = img_as_ubyte(out_img)
imsave('examples/seasons/out_img.png', img2)

imshow(img2)
#freq_l = [] # frequencies for each layer of laplas pyramid
#freq_g = [] # frequencies for each layer of gauss pyramid

#for i in range(0, layers):
#    freq_l.append(np.log(1 + abs(scipy.fft.fftshift(scipy.fft.fft2(l[i])))))
#    freq_g.append(np.log(1 + abs(scipy.fft.fftshift(scipy.fft.fft2(g[i])))))

#for i in range(0, layers):
#    imsave('freq_l' + str(i+1) + '.png', freq_l[i])
#    imsave('freq_g' + str(i+1) + '.png', freq_g[i])
#    imsave('out_l' + str(i+1) + '.png', l[i])
#    imsave('out_g' + str(i+1) + '.png', g[i])
