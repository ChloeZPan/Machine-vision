import numpy as np
import cv2


class CV249:
    def cvt_to_gray(self, img):
        # Note that cv2.imread will read the image to BGR space rather than RGB space

        # TODO: your implementation
        dst = np.dot(img[..., :3], [0.114, 0.587, 0.299]).round()
        return dst

    def blur(self, img, kernel_size=(3, 3)):
        """smooth the image with box filter
        
        Arguments:
            img {np.array} -- input array
        
        Keyword Arguments:
            kernel_size {tuple} -- kernel size (default: {(3, 3)})
        
        Returns:
            np.array -- blurred image
        """
        # TODO: your implementation
        kernel = np.ones(kernel_size, np.float32)/(kernel_size[0]*kernel_size[1])
        return cv2.filter2D(img, -1, kernel)

    def sharpen_laplacian(self, img):
        """sharpen the image with laplacian filter
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            np.array -- sharpened image
        """

        # subtract the laplacian from the original image 
        # when have a negative center in the laplacian kernel

        # TODO: your implementation
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian = cv2.filter2D(img, -1, kernel)
        sharpened = np.subtract(img, laplacian)
        return sharpened

    def unsharp_masking(self, img):
        """sharpen the image via unsharp masking
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            np.array -- sharpened image
        """
        # use don't use cv2 in this function
        
        # TODO: your implementation
        mask = np.subtract(img, self.blur(img))
        return np.add(img, mask)

    def edge_det_sobel(self, img):
        """detect edges with sobel filter
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            [np.array] -- edges
        """

        # TODO: your implementation
        x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kx_i = cv2.filter2D(img, -1, x)
        ky_i = cv2.filter2D(img, -1, y)
        edges = np.sqrt(kx_i ** 2 + ky_i ** 2).astype(np.uint8)
        return edges
