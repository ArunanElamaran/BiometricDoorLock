from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import time

def verify(img1_path, img2_path, displayImages = False, model_name='VGG-Face'):

    if displayImages:
        # Read in the images for to display them
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        plt.imshow(img1[:, :, ::-1])
        plt.show()
        plt.imshow(img2[:, :, ::-1])
        plt.show()

    start_time = time.perf_counter()
    result = DeepFace.verify(img1_path, img2_path, model_name=model_name)
    end_time = time.perf_counter()
    print(f"Model: {model_name}\nResult: {result}\nTime taken: {end_time - start_time}\n\n") 
    # Positive verification if distance is less than max threshold

if __name__ == "__main__":
    overall_path = "database/Max/"
    img1_path = overall_path + "img1.jpg"
    img2_path = overall_path + "img3.jpg"
    model_types = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID']

    for model in model_types:
        verify(img1_path, img2_path, displayImages=False, model_name=model)