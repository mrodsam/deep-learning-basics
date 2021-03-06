import glob
import cv2
import numpy as np
import csv

from train import create_model, IMAGE_SIZE
from keras.applications.mobilenetv2 import preprocess_input

WEIGHTS_FILE = "model-0.53.h5"
IMAGES = "REY_scan_anonim/*"
csv_file="validation.csv"

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)
    filenames_test = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            name = row[2].split(":")[1]
            filenames_test.append(name)
    i = 0
    for filename in glob.glob(IMAGES):
        if filename.split("/")[1] in filenames_test:
            i+=1
            unscaled = cv2.imread(filename)
            image_height, image_width, _ = unscaled.shape
    
            image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
            feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
    
            region = model.predict(x=np.array([feat_scaled]))[0]
    
            x0 = int(region[0] * image_width / IMAGE_SIZE)
            y0 = int(region[1] * image_height / IMAGE_SIZE)
    
            x1 = int((region[0] + region[2]) * image_width / IMAGE_SIZE)
            y1 = int((region[1] + region[3]) * image_height / IMAGE_SIZE)
    
            cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.imwrite('inference_images/image'+str(i)+'.png', unscaled)
            cv2.imshow("image", unscaled)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
            

if __name__ == "__main__":
    main()