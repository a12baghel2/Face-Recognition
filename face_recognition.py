import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

#Load the haarcascade classifier file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
data_set = "./"

#Enter the filename or the name of person you are collecting the data 
file_name = input("Enter the File Name: ")

#main

while True:
	ret,frame = cap.read()
	#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key= lambda f:f[2]*f[3])

	#Drawing Rectangle around the Face
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,230,255),2)

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))

	#cv2.imshow("face section",face_section)

	#Showing the image
	cv2.imshow("large_frame",frame)

	#Exit condition. When 'q' is pressed
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

#creating the data array that stores the data 
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#saving in npy file in the directory
np.save(data_set+file_name+".npy",face_data)
print("data successfully saved")


cap.release()
cv2.destroyAllWindows()
