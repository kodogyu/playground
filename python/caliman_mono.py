# https://github.com/astar-ai/calicam_mono/blob/master/calicam_mono.py


import sys
import cv2
import math
import numpy as np

#-------------------------------------------------------------------------------#

Kl = None
Dl = None
xil = None

mapx = None
mapy = None

cap_cols = None
cap_rows = None

vfov_now = 120
width_now = 640

changed = False

margin = 15. * math.pi / 180.

mode = 'kRectPerspective'

#-------------------------------------------------------------------------------#

def load_parameters(param_file):
  global Kl, Dl, xil
  global cap_cols, cap_rows
  
  # fs = cv2.FileStorage(param_file, cv2.FILE_STORAGE_READ)
  
  # cap_size_node = fs.getNode("cap_size")
  # cap_cols = int(cap_size_node.at(0).real())
  # cap_rows = int(cap_size_node.at(1).real())

  # Kl = fs.getNode("Kl").mat()
  # Dl = fs.getNode("Dl").mat()
  # xil = fs.getNode("xil").mat()

  # KITTI-360 camera2 intrinsic parameters (datasets/KITTI-360/calibration/image_02.yaml)
  cap_size_node = (1400, 1400)
  cap_cols = int(cap_size_node[0])
  cap_rows = int(cap_size_node[1])
  xi = 2.2134047507854890e+00
  # xi = 1.0
  k1, k2 = 1.6798235660113681e-02, 1.6548773243373522e+00
  p1, p2 = 4.2223943394772046e-04, 4.2462134260997584e-04
  gamma1, gamma2 = 1.3363220825849971e+03, 1.3357883350012958e+03
  u0, v0 = 7.1694323510126321e+02, 7.0576498308221585e+02

  k360_K = np.array([[gamma1,      0, u0],
                    [     0, gamma2, v0],
                    [     0,      0, 1]])
  k360_D = np.array([[k1, k2, p1, p2]])
  k360_xi = np.array([xi])

  Kl = k360_K
  Dl = k360_D
  xil = xi
  
#-------------------------------------------------------------------------------#

def init_undistort_rectify_map(k, d, r, knew, xi0, size, mode):
  fx = k[0, 0]
  fy = k[1, 1]
  cx = k[0, 2]
  cy = k[1, 2]
  s  = k[0, 1]
  
  k1 = d[0, 0]
  k2 = d[0, 1]
  p1 = d[0, 2]
  p2 = d[0, 3]
  
  ki = np.linalg.inv(knew)
  ri = np.linalg.inv(r)
  kri = np.linalg.inv(np.matmul(knew, r))
  
  rows = size[0]
  cols = size[1]
  
  mapx = np.zeros((rows, cols), dtype = np.float32)
  mapy = np.zeros((rows, cols), dtype = np.float32)
  
  print("Wait, this takes a while ... ")
  for r in range(rows):
    for c in range(cols):
      xc = 0.0
      yc = 0.0
      zc = 0.0
      
      if mode == 'kRectPerspective':
        cr1 = np.array([c, r, 1.])
        xc = np.dot(kri[0, :], cr1)
        yc = np.dot(kri[1, :], cr1)
        zc = np.dot(kri[2, :], cr1)
  
      if mode == 'kRectLonglat':
        tt = (c * 1. / (cols - 1) - 0.5) * math.pi
        pp = (r * 1. / (rows - 1) - 0.5) * math.pi

        xn = math.sin(tt)
        yn = math.cos(tt) * math.sin(pp)
        zn = math.cos(tt) * math.cos(pp)
      
        cr1 = np.array([xn, yn, zn])
        xc = np.dot(ri[0, :], cr1)
        yc = np.dot(ri[1, :], cr1)
        zc = np.dot(ri[2, :], cr1)
  
      if mode == 'kRectFisheye':
        cr1 = np.array([c, r, 1.])
        ee = np.dot(ki[0, :], cr1)
        ff = np.dot(ki[1, :], cr1)
        zz = 2. / (ee * ee + ff * ff + 1.)

        xn = zz * ee
        yn = zz * ff
        zn = zz - 1.

        cr1 = np.array([xn, yn, zn])
        xc = np.dot(ri[0, :], cr1)
        yc = np.dot(ri[1, :], cr1)
        zc = np.dot(ri[2, :], cr1)
        
      if mode == 'kRectCylindrical':
        cr1 = np.array([c, r, 1.])
        tt = np.dot(ki[0, :], cr1)
        pp = np.dot(ki[1, :], cr1) + margin

        xc = -math.sin(pp) * math.cos(tt)
        yc = -math.cos(pp)
        zc =  math.sin(pp) * math.sin(tt)
        
      if zc < 0.0:
        mapx[r, c] = np.float32(-1.)
        mapy[r, c] = np.float32(-1.)

        continue
      
      rr = math.sqrt(xc * xc + yc * yc + zc * zc)
      xs = xc / rr
      ys = yc / rr
      zs = zc / rr
      
      xu = xs / (zs + xi0)
      yu = ys / (zs + xi0)
      
      r2 = xu * xu + yu * yu
      r4 = r2 * r2
      xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
      yd = (1 + k1 * r2 + k2 * r4) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu)

      u = fx * xd + s * yd + cx
      v = fy * yd + cy

      mapx[r, c] = np.float32(u)
      mapy[r, c] = np.float32(v)

  return mapx, mapy

#-------------------------------------------------------------------------------#

def init_rectify_map():
  global mode, mapx, mapy

  Rl = np.identity(3, dtype = np.float64)
  Knew = None
  
  if mode == 'kRectPerspective':
    print('kRectPerspective')

    vfov_rad = vfov_now * math.pi / 180.
    focal = width_now / 2. / math.tan(vfov_rad / 2.)
    
    Knew = np.identity(3, dtype = np.float64)
    Knew[0, 0] = focal
    Knew[1, 1] = focal
    Knew[0, 2] = width_now / 2 - 0.5
    Knew[1, 2] = width_now / 2 - 0.5

    img_size = [width_now, width_now]
    mapx, mapy = init_undistort_rectify_map(Kl, Dl, Rl, Knew, xil, img_size, 'kRectPerspective')
  
    print('Width: {}, Height: {}, V.FoV: {}'.format(width_now, width_now, vfov_now))
    
  if mode == 'kRectLonglat':
    print('kRectLonglat')

    Knew = np.identity(3, dtype = np.float64)
    Knew[0, 0] = width_now / math.pi
    Knew[1, 1] = width_now / math.pi
    
    img_size = [width_now, width_now]
    mapx, mapy = init_undistort_rectify_map(Kl, Dl, Rl, Knew, xil, img_size, 'kRectLonglat')

    print('Width: {}, Height: {}'.format(width_now, width_now))
    
  if mode == 'kRectFisheye':
    print('kRectFisheye')

    Knew = np.identity(3, dtype = np.float64)
    Knew[0, 0] = width_now / 2
    Knew[1, 1] = width_now / 2
    Knew[0, 2] = width_now / 2 - 0.5
    Knew[1, 2] = width_now / 2 - 0.5

    img_size = [width_now, width_now]
    mapx, mapy = init_undistort_rectify_map(Kl, Dl, Rl, Knew, xil, img_size, 'kRectFisheye')

    print('Width: {}, Height: {}'.format(width_now, width_now))
    
  if mode == 'kRectCylindrical':
    print('kRectCylindrical')

    Knew = np.identity(3, dtype = np.float64)

    Knew[0, 0] = width_now / math.pi
    Knew[1, 1] = width_now / (math.pi - 2 * margin)
    
    img_size = [width_now, width_now]
    mapx, mapy = init_undistort_rectify_map(Kl, Dl, Rl, Knew, xil, img_size, 'kRectCylindrical')

    print('Width: {}, Height: {}'.format(width_now, width_now))
    
  print('K Matrix:')
  print(Knew)
  print('')
  
#-------------------------------------------------------------------------------#

def main():  
  global changed, mode, mapx, mapy
  
  param_file = "astar_calicam_mono.yml"
  # image_name = "times_square.jpg"
  image_name = "../playground_shared/datasets/KITTI-360/data/2013_05_28_drive_0000_sync/image_02/0000000009.png"

  if len(sys.argv) == 2:
    param_file = sys.argv[1]

  if len(sys.argv) == 3:
    param_file = sys.argv[1]
    image_name = sys.argv[2]
  
  load_parameters(param_file)
  init_rectify_map()
  
  raw_img = cv2.imread(image_name, 1)
  
  param_win_name = "Raw Image: " + str(cap_cols) + " x " + str(cap_rows)
  
  cv2.namedWindow(param_win_name)
 
  while True:
    if changed == True:
      init_rectify_map()
      changed = False
    
    raw_imgl = raw_img
    rect_imgl = cv2.remap(raw_imgl, mapx, mapy, cv2.INTER_LINEAR)
    
    dim = (int(cap_cols / 2), int(cap_rows / 2))
    small_img = cv2.resize(raw_imgl, dim, interpolation = cv2.INTER_NEAREST)
    
    cv2.imshow(param_win_name, small_img)
    cv2.imshow("Rectified Image", rect_imgl)
    
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('1'):
      mode = 'kRectPerspective'
      changed = True
    
    if key & 0xFF == ord('2'):
      mode = 'kRectCylindrical'
      changed = True
    
    if key & 0xFF == ord('3'):
      mode = 'kRectFisheye'
      changed = True
    
    if key & 0xFF == ord('4'):
      mode = 'kRectLonglat'
      changed = True
    
    if key & 0xFF == ord('q') or key  == 27:
      break

#-------------------------------------------------------------------------------#

if __name__ == "__main__":
  main()

#-------------------------------------------------------------------------------#