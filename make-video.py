import os
import cv2

def images_to_video(image_folder, output_video):
    # Get all PNG files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort images alphabetically
    print(len(images))
    # Determine the dimensions of the images
    img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = img.shape

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    video = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))  # 10 fps

    # Write images to video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release VideoWriter object
    video.release()

if __name__ == "__main__":
    # Specify the folder containing PNG images and the output video file
    image_folder = "images-selected"
    output_video = "./output.mp4"

    # Convert images to video
    images_to_video(image_folder, output_video)
