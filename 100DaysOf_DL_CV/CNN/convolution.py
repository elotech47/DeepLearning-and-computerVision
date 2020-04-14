from skimage.exposure import rescale_intensity
import numpy as np 
import argparse
import cv2


#Defining the convolve method
def convolve(image, K):
    #getting the spatial dimension of the image and kener
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    #set a memory for the output image, 
    #set padding to maintain spatial size of the imahe
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
            cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    #looping over the input image

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):

          # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

        #performing the actual convolution by taking
        #element-wise multiplication btw ROI and 
        #the kernel, then summing the matrix

            k = (roi * K).sum()

        #store the convolved value in the output (x,y) - coordinate of the output image
            output[y-pad, x-pad] = k

        #rescale the output image to be in the range of (0,255)
            output = rescale_intensity(output, in_range= (0,255))
            output = (output * 255).astype("uint8")

        return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
args = vars(ap.parse_args())


#construct average blurring kernels used to smooth an image
smallBlur = np.ones((7,7), dtype = "float")*(1.0/(7*7))
largeBlur = np.ones((21,21), dtype = "float")*(1.0/(21*21))

#sharpening filter
sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]), dtype="int")

#laplacian kernel to detect edge
laplacian = np.array(([0,1,0],
                    [1,-4,1],
                    [0,1,0]), dtype="int")

#Sobel kernels used to detect edges in the x and y direction
#Sobel kernel for x-axis
sobelX = np.array(([-1,0,1],
                    [-2,0,2],
                    [-1,0,1]), dtype="int")

#Sobel for y-axis
sobelY = np.array(([-1,-2,-1],
                    [0,0,0],
                    [1,2,1]), dtype="int")

#emboss kernel
emboss = np.array(([-2,-1,0],
                    [-1,1,1],
                    [0,1,2]), dtype="int")

kernelBank = (
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY),
        ("emboss", emboss)
)


#load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#loop over the kernel bank
for (kernelName, K) in kernelBank:
    #apply kernel to grayscale image
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    #shpw the output image
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - convolve".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()