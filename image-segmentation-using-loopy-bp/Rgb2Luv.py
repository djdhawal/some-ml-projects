from __future__ import print_function

import numpy as np
import skimage


class Rgb2Luv(object):

    # Convert an image in RGB colorspace to CIE-LUV colorspace as
    #     RGB --> XYZ --> LUV
    def convert(self, img):

        XYZ = np.array([[0.4125, 0.3576, 0.1804], [0.2125, 0.7154, 0.0721],
                        [0.0193, 0.1192, 0.9502]])
        XYZ = np.float32(XYZ)
        Yn = 1.0
        Lt = 0.008856
        Up = 0.19784977571475
        Vp = 0.46834507665248

        imgshape = np.shape(img)

        # If the image was a PNG, it may have a depth of 4
        if imgshape[2] == 4:
            img = img[:, :, 0:-1]
            imgshape = np.shape(img)
        elif imgshape[2] != 3:
            print('Image must have three color channels')
            return None

        if not isinstance(img, float):
            img = skimage.img_as_float(img)
            img = np.float32(img)

        img = np.transpose(img, (2, 0, 1))
        img = np.reshape(img, (3, imgshape[0]*imgshape[1]), 'F')

        # Convert RGB to XYZ color space
        xyz = XYZ.dot(img).transpose()
        xyz = np.reshape(xyz, (imgshape[0], imgshape[1], -1), 'F')
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

        # Convert XYZ to LUV color space
        l0 = y/Yn
        l = np.copy(l0)

        l[l0 > Lt] = 116.0*(l0[l0 > Lt]**(1.0/3.0)) - 16.0
        l[l0 <= Lt] = 903.3*l0[l0 <= Lt]
        c = x + 15.0*y + 3.0 * z
        u = 4.0*np.ones([imgshape[0], imgshape[1]], img.dtype)
        v = (9.0/15.0)*np.ones([imgshape[0], imgshape[1]], img.dtype)
        u[c != 0] = 4.0*x[c != 0]/c[c != 0]
        v[c != 0] = 9.0*y[c != 0]/c[c != 0]

        u = 13.0*l*(u-Up)
        v = 13.0*l*(v-Vp)

        luvimg = np.array([l, u, v])
        luvimg = luvimg.transpose(1, 2, 0)

        return luvimg
