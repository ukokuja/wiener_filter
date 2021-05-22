import os
import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

RESULT_FILENAME_KEY = "{folder}/result_{{quality}}.jpg"
REGION_SIZE = 3


def best_cmp_fn(a, b):
    return a < b


def worse_cmp_fn(a, b):
    return best_cmp_fn(b, a)


class Noiser():
    def blur(self, img, sigma_blur):
        """
        :param img: cv2 image instance
        :param sigma_blur: int indicating blur quantity
        :return: image with gaussian blur
        """
        return cv2.GaussianBlur(img, (REGION_SIZE, REGION_SIZE), sigma_blur)

    def add_gaussian_noise(self, img, sigma_noise):
        """

        :param img: cv2 image instance
        :param sigma_noise: int indicating blur quantity
        :return: image with gaussian noise
        """
        gauss = np.random.normal(0, sigma_noise, np.shape(img))
        noisy_img = img + gauss
        noisy_img[noisy_img < 0] = 0
        noisy_img[noisy_img > 255] = 255
        return noisy_img

    def run(self, filename, sigma_noise, sigma_blur):
        """
        :param filename: image filename
        :param sigma_noise: int indicating gaussian noise quantity
        :param sigma_blur: int indicating gaussian blur quantity
        :return: noised image and folder location where it saved the modified image
        """
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        blur_img = self.blur(img, sigma_blur)
        noise_img = self.add_gaussian_noise(blur_img, sigma_noise)

        _folder = 'results_{sigma_noise}_{sigma_blur}'.format(sigma_noise=sigma_noise, sigma_blur=sigma_blur)
        self._create_folder(_folder)
        cv2.imwrite('{folder}/image_with_noise.png'.format(folder=_folder), noise_img)
        return noise_img, _folder

    def _create_folder(self, folder):
        try:
            os.mkdir(folder)
        except:
            pass


class Wiener():

    def degradation_function(self, sigma_noise, region_size=REGION_SIZE):
        """

        :param sigma_noise: noise added to the img
        :param region_size: size of the window where noise was added
        :return:
        """
        h = gaussian(region_size, sigma_noise).reshape(region_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def run(self, img, sigma_noise, output_filename):
        """
        The function finds best results and output the image
        :param img: cv2 image instance
        :param sigma_noise: int indicating gaussian noise quantity
        :param output_filename: str indicating output filename

        """
        H = self.degradation_function(sigma_noise, region_size=REGION_SIZE)
        H /= np.sum(H)
        G = fft2(img)
        H = fft2(H, s=G.shape)
        H_star = np.conj(H)
        H_2 = np.abs(H) ** 2

        self._calculate_best_k_and_output_image(G, H_2, H_star, output_filename, img, cmp_fn=best_cmp_fn, quality="best")
        self._calculate_best_k_and_output_image(G, H_2, H_star, output_filename, img, cmp_fn=worse_cmp_fn, quality="worse")

    def _calculate_best_k_and_output_image(self, G, H_2, H_star, filename, img, cmp_fn, quality):
        """
        The method gets the best K and outputs the image
        """
        best_k, F_hat = self._find_K_by_function(G, H_2, H_star, img, cmp_fn, key=quality)
        plt.imshow(F_hat, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title("{} K={}".format(quality, best_k))
        plt.savefig(filename.format(quality=quality), bbox_inches='tight')

    def _find_best_K_by_function(self, G, H_2, H_star, img, cmp_fn, key):
        """
        Iterates over every possible K, calculates the error and chooses the best one
        """
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
            best_error, best_k, best_F_hat = (error, k, F_hat) if cmp_fn(error, best_error) else (
                best_error, best_k, best_F_hat)
        print("{}_error={}".format(key, best_error))
        print("{}_k={}".format(key, best_k))
        return best_k, best_F_hat

    def _get_F_hat(self, H_2, H_star, G, k):
        """

        :param H_2: square of is the magnitude of the Fourier transform of H
        :param H_star: complex conjugate of the Fourier transform of H
        :param G: Fourier transform of g
        :param k: constant related to the noise,
        :return: F_Hat based on the wiener formula
        """
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
        image_with_noise, folder = noiser.run(filename, **case)
        wiener.run(image_with_noise, sigma_noise=case['sigma_noise'],
                   output_filename=RESULT_FILENAME_KEY.format(folder=folder))
