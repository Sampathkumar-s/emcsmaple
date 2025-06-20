import cv2
import face_recognition
import numpy as np

# Load known face images
known_image_1 = face_recognition.load_image_file("person1.jpg")
known_image_2 = face_recognition.load_image_file("person2.jpg")

# Encode faces
known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]

# Store encodings and labels
known_encodings = [known_encoding_1, known_encoding_2]
known_names = ["Person 1", "Person 2"]

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        
        # Check for best match
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_names[best_match_index]
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
