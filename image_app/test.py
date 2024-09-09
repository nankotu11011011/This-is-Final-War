from pydrive_2 import GoogleDriveAPI

drive = GoogleDriveAPI()

drive.download_from_file_name("models","face_detection_model.h5","./models")