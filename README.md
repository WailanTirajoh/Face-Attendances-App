# Face Attendance App

## For Windows User Installation
- [Download Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Windows-x86_64.exe)
- Install miniconda and open the anaconda prompt
- Inside anaconda prompt write commands:
    - conda create -n FaceAttendances python=3.6
        - Y
        - Y
    - activate FaceAttendances
        - CD to face attendance app:
            - pip install -r requirements.txt
            - Python main.py
## How to use?
- ### Create User
    - Creator
    - Fill the form
    - fill Position ID 1 for Student
    - Snap your face with cam, Press Q
    - Done, User and face saved

- ### Trainer
    - Align Face
    - Preprocess
    - Train Images
    - Add Photo if you want the user have more than 1 photo (Optional)

- ### Database
    - Username: admin
    - Password: admin
    - Sub menu people for list of people in database
    - Sub menu position for add more position to database

- ### Detector
    - LiveCam 
        - created for testing purposes
    - Record
        - for record something in cam
    - Detect Record 
        - for detect any face from last record
    - Upload
        - for upload any video to be detect the face
    - UploadSave
        - for upload any video, detect, and save
    - Attendance
        - Main procedure for this app
    - Gather Liveness
        - Upload face and detect if its fake or real and save to folder 
