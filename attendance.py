import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# -------------------
# Load known faces
# -------------------
known_faces_data = {
    "Subhanshu": "faces/subh.jpg",
    "Naitik": "faces/naitik.jpg",
    "Anand": "faces/anand.jpg",
    "Ganesh": "faces/ganesh.jpg",
    "Suyash": "faces/suyash.jpg"
}

known_faces_encoding = []
known_faces_name = []

for name, path in known_faces_data.items():
    try:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces_encoding.append(encoding)
        known_faces_name.append(name)
    except IndexError:
        print(f"⚠️ No face found in the image for {name} at {path}")
    except FileNotFoundError:
        print(f"⚠️ Image file not found for {name} at {path}")

# -------------------
# Attendance CSV setup
# -------------------
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
attendance_logged = set()

csv_file = open(f"{current_date}.csv", "a+", newline="")
lnwriter = csv.writer(csv_file)
if csv_file.tell() == 0:  # Write header if file is empty
    lnwriter.writerow(["Name", "Time"])

# -------------------
# Video + Tkinter Setup
# -------------------
video_capture = None
process_this_frame = True
running = False

def start_attendance():
    global video_capture, running
    if running:
        return
    running = True
    video_capture = cv2.VideoCapture(0)
    update_frame()

def stop_attendance():
    global video_capture, running
    running = False
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()
    csv_file.flush()
    messagebox.showinfo("Stopped", "Attendance system stopped")

def update_frame():
    global process_this_frame, video_capture, running

    if not running:
        return

    ret, frame = video_capture.read()
    if not ret:
        return

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_name[best_match_index]

            # Draw box
            top, right, bottom, left = [v * 4 for v in face_location]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (255, 255, 255), 1)

            # Log attendance once
            if name in known_faces_name and name not in attendance_logged:
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])
                csv_file.flush()
                attendance_logged.add(name)
                print(f"✅ Attendance logged for {name} at {current_time}")

                # Insert into GUI table
                attendance_table.insert("", "end", values=(name, current_time))

    process_this_frame = not process_this_frame

    # Convert frame to display in Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

# -------------------
# Tkinter GUI
# -------------------
root = tk.Tk()
root.title("Face Recognition Attendance System")

# Live video feed
video_label = tk.Label(root)
video_label.pack()

# Start & Stop buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

btn_start = tk.Button(btn_frame, text="Start Attendance", command=start_attendance,
                      bg="green", fg="white", font=("Arial", 12), width=15)
btn_start.grid(row=0, column=0, padx=10)

btn_stop = tk.Button(btn_frame, text="Stop Attendance", command=stop_attendance,
                     bg="red", fg="white", font=("Arial", 12), width=15)
btn_stop.grid(row=0, column=1, padx=10)

# Attendance Table
table_frame = tk.Frame(root)
table_frame.pack(pady=10)

attendance_table = ttk.Treeview(table_frame, columns=("Name", "Time"), show="headings", height=8)
attendance_table.heading("Name", text="Name")
attendance_table.heading("Time", text="Time")
attendance_table.column("Name", width=150)
attendance_table.column("Time", width=100)
attendance_table.pack()

# Handle close button
root.protocol("WM_DELETE_WINDOW", stop_attendance)

root.mainloop()

# Cleanup
csv_file.close()
