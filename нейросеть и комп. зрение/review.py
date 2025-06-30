import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt


def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros_like(image)

    for contour in contours:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_val = cv2.mean(gray, mask=mask)[0]

        color = (int(mean_val), 0, 255 - int(mean_val))  # RGB
        cv2.drawContours(contour_image, [contour], -1, color, thickness=2)

    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main():

    Tk().withdraw()
    file_path = askopenfilename(title="Выберите изображение", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        process_image(file_path)
    else:
        print("Файл не выбран.")


if __name__ == "__main__":
    main()
