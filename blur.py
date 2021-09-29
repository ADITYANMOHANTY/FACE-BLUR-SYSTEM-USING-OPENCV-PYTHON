import cv2

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_and_blur(bw, color): 
    # detect al faces
    faces = cascade.detectMultiScale(bw, 1.1, 4)
    # get the locations of the faces
    for (x, y, w, h) in faces:
        # select the areas where the face was found
        roi_color = color[y:y+h, x:x+w]
        # blur the colored image
        blur = cv2.GaussianBlur(roi_color, (101,101), 0)
        # Insert ROI back into image
        color[y:y+h, x:x+w] = blur            
    
    # return the blurred image
    return color

# turn camera on
image_capture = cv2.imread('test.jpg')

while True:
    # get the picture
    
    # transform color -> grayscale
    bw = cv2.cvtColor(image_capture, cv2.COLOR_BGR2GRAY)
    # detect the face and blur it
    blur = find_and_blur(bw, image_capture)
    # display the blur output
    cv2.imshow('Image', blur)
    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# turn camera off        
image_capture.release()
cv2.waitKey()
 