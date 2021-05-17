import os

import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian

RESULT_FILENAME_KEY = "results_{sigma_noise}_{sigma_blur}/result_{{quality}}.jpg"
import matplotlib.pyplot as plt

REGION_SIZE = 3

def best_cmp_fn(a, b):
    return a < b

def worse_cmp_fn(a, b):
    return best_cmp_fn(b, a)

class Noiser():
    def blur(self, img, sigma_blur):
        img_copy = np.copy(img)
        h = np.eye(sigma_blur) / sigma_blur
        img_copy = convolve2d(img_copy, h, mode='valid')
        return img_copy

    def add_gaussian_noise(self, img, sigma_noise):
        gauss = np.random.normal(0, sigma_noise, np.shape(img))
        noisy_img = img + gauss
        noisy_img[noisy_img < 0] = 0
        noisy_img[noisy_img > 255] = 255
        return noisy_img

    def run(self, filename, sigma_noise, sigma_blur):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        blur_img = cv2.GaussianBlur(img, (5, 5), sigma_blur)
        noise_img = self.add_gaussian_noise(blur_img, sigma_noise)
        # Display the noise image

        folder = 'results_{sigma_noise}_{sigma_blur}'.format(sigma_noise=sigma_noise, sigma_blur=sigma_blur)
        self._create_folder(folder)
        cv2.imwrite('{folder}/image_with_noise.png'.format(folder=folder), noise_img)
        return noise_img

    def _create_folder(self, folder):
        try:
            os.mkdir(folder)
        except:
            pass


class Wiener():

    def degradation_function(self, region_size=REGION_SIZE):
        h = gaussian(region_size, region_size / 3).reshape(region_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def run(self, img, filename):
        H = self.degradation_function(region_size=REGION_SIZE)
        H /= np.sum(H)
        G = fft2(img)
        H = fft2(H, s=G.shape)
        H_star = np.conj(H)
        H_2 = np.abs(H) ** 2
        best_k, best_F_hat = self.get_cmp_values(G, H_2, H_star, img, best_cmp_fn, key='best')
        plt.imshow(best_F_hat, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title("best K={}".format(best_k))
        plt.savefig(filename.format(quality='best'), bbox_inches='tight')

        worse_k, worse_F_hat = self.get_cmp_values(G, H_2, H_star, img, worse_cmp_fn, key='worse')
        plt.imshow(worse_F_hat, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title("worse K={}".format(worse_k))
        plt.savefig(filename.format(quality='worse'), bbox_inches='tight')

    def get_cmp_values(self, G, H_2, H_star, img, cmp_fn, key):
        best_error = -1
        best_k = -1
        start = 10
        best_F_hat = None
        for _k in range(start, 2000):
            k = _k / 1000
            F_hat = self._get_F_hat(H_2, H_star, G, k)
            error = ((img - F_hat) ** 2).mean()
            if _k == start:
                best_error = error
                best_k = k
            best_error, best_k, best_F_hat = (error, k, F_hat) if cmp_fn(error, best_error) else (best_error, best_k, best_F_hat)
        print("{}_error={}".format(key, best_error))
        print("{}_k={}".format(key, best_k))
        return best_k, best_F_hat


    def _get_F_hat(self, H_2, H_star, G, k):
        H = H_star / (H_2 + k)
        G = G * H
        inverse_fourier = np.abs(ifft2(G))
        return inverse_fourier



if __name__ == "__main__":
    noiser = Noiser()
    wiener = Wiener()

    filename = "lena.png"
    cases = [{
        "sigma_noise": 1,
        "sigma_blur": 1.5,
    }, {
        "sigma_noise": 5,
        "sigma_blur": 1.5,
    }, {
        "sigma_noise": 15,
        "sigma_blur": 1.5,
    }]

    for case in cases:
        image_with_noise = noiser.run(filename, **case)
        wiener.run(image_with_noise, filename=RESULT_FILENAME_KEY.format(**case))
