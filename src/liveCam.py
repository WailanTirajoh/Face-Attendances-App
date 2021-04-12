import tensorflow as tf
import numpy as np
import src.detect_face
import cv2,os
import src.facenet
import pickle
import time
import math
import sqlite3
import winsound
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random
import string
stderr = sys.stderr										#ini buat hilangkan using tensorflow backend
sys.stderr = open(os.devnull, 'w')						#ini buat hilangkan using tensorflow backend
from collections import Counter
from scipy import misc
from datetime import datetime
from datetime import timedelta
from statistics import mean
from operator import itemgetter
from tkinter import messagebox
from keras.models import load_model
from keras.preprocessing.image import img_to_array

sys.stderr = stderr										#ini buat hilangkan using tensorflow backend

conn = sqlite3.connect('Facebase.db')
c = conn.cursor()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	#mematikan warning tensorflow

STD_COLORS = {
# Format BGR (Blue, Green, Red)
    'green': (0,255,0),
    'blue': (255,0,0),
    'red': (0,0,255),
    'black': (0,0,0),
    'white': (255,255,255),
    'gray': (128,128,128),
    'magneta': (255,0,255),
    'Green': (0,255,0),
    'Blue': (255,0,0),
    'Red': (0,0,255),
    'Black': (0,0,0),
    'White': (255,255,255),
    'Gray': (128,128,128),
    'Magneta': (255,0,255),
}

STD_SESI = {
	"1":("07:00:00"),
	"2":("10:00:00"),
	"3":("13:00:00"),
	"4":("16:00:00"),
	"maxTelat":("0:15:00"),
}

# chart yang akan ditampilkan setelah uji deteksi video berakhir.
def chart(theframe,theaccuracy,nameMF,frameMF,newDist,newAcc):
	plt.figure('Graph Report')
	plt.title('Graph')
	plt.style.use("ggplot")	

	plt.subplot(2, 1, 2)
	plt.plot(theframe,theaccuracy)
	plt.ylabel('Accuracy (%)')
	plt.xlabel('Frames')
	
	plt.subplot(2, 2, 1)
	plt.bar(nameMF,frameMF)
	plt.ylabel('Frames')
	plt.xlabel('Name')
	
	plt.subplot(2,2,2)
	plt.plot(newDist, newAcc)
	plt.xlabel('Distance (Cm)')
	plt.ylabel('Accuracy (%)')

	plt.show()
	plt.savefig("LivePlot.png")



# function ambil data diri dari database sqlite
def getProfile(id):
    conn = sqlite3.connect("Facebase.db")
    query = "SELECT * FROM Peoples join Positions using('PositionID') WHERE PeopleID="+str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

# function konversi waktu 
def convertTime(times):
	ftr = [3600, 60, 1]
	reTime = sum([a*b for a,b in zip(ftr,map(int,times.split(':')))])
	return reTime

# function untuk konversi waktu saat ini menjadi waktu saat ini, dan status, dan sesi dipakai untuk presensi
def treshHours(times):
	convertSesi1 = convertTime(STD_SESI['1'])
	convertSesi2 = convertTime(STD_SESI['2'])
	convertSesi3 = convertTime(STD_SESI['3'])
	convertSesi4 = convertTime(STD_SESI['4'])
	convertHours = convertTime(times)
	maxTelat = convertTime(STD_SESI['maxTelat'])
	banding1 = convertHours - (convertSesi1)
	banding2 = convertHours - (convertSesi2)
	banding3 = convertHours - (convertSesi3)
	banding4 = convertHours - (convertSesi4)
	telat = min(banding1,banding2,banding3,banding4, key=abs)
	sesi = ("Sesi 1", "Sesi 2", "Sesi 3", "Sesi 4")
	l = (banding1, banding2, banding3, banding4)
	X = l.index(min(l, key=abs))
	sesis = sesi[X]
	if telat > maxTelat:
		waktu = time.strftime('%H:%M:%S', time.gmtime(int(telat)))
		statusWaktu = "Terlalu telat"
	elif telat > 0:
		waktu = time.strftime('%H:%M:%S', time.gmtime(int(telat)))
		statusWaktu = "telat"
	elif telat <= 0:
		waktu = time.strftime('%H:%M:%S', time.gmtime(abs(int(telat))))
		statusWaktu = "Cepat"
	return waktu,statusWaktu, sesis

# function untuk mencari nama dari folder dalam pre_img, dipakai untuk mendapatkan nama dari index setelah train classifier karena classifier hanya mengembalikan index
def cariNama():
	names = []
	for root, dirs, files in os.walk("./images/3pre_img"):
		for filename in dirs:
			names.append(filename)
	return names

def randomString(stringLength):
	letters = string.ascii_letters
	return ''.join(random.choice(letters) for i in range(stringLength))

# prosedur non parameter untuk record menggunakan webcam.
def recordCam():
	STD_DIMENSIONS = {
		"480p": (640, 480),
		"720p": (1280,720),
		"1080p": (1920, 1080),
		"4k": (3840, 2160)
	}
	i = 0
	while True:
		if  os.path.exists("record\\video"+str(i)+".mp4") == True:
			i = i+1
		else:
			filename = "record\\video"+str(i)+".mp4"
			break
	frames_per_seconds = 10.0
	my_res = '720p'
	def change_res(cam, width, height):
		cam.set(3, width)
		cam.set(4, height)
	def get_dims(cam, res='480p'):
		width, height = STD_DIMENSIONS['480p']
		if res in STD_DIMENSIONS:
			width, height = STD_DIMENSIONS[res]
		change_res(cam, width,height)
		return width, height
	VIDEO_TYPE = {
		'avi': cv2.VideoWriter_fourcc(*'XVID'),
		'mp4': cv2.VideoWriter_fourcc(*'mp4v')
	}
	def get_video_type(filename):
		filename, ext = os.path.splitext(filename)
		if ext in VIDEO_TYPE:
			return VIDEO_TYPE[ext]
		return VIDEO_TYPE['mp4']
	cam = cv2.VideoCapture(0)
	dims = get_dims(cam, res=my_res)
	video_type_cv2 = get_video_type(filename)
	out = cv2.VideoWriter(filename, video_type_cv2, frames_per_seconds, dims)
	while(True):
		ret, frame = cam.read()
		out.write(frame)
		cv2.imshow('Recording...',frame)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			break
	cam.release()
	out.release()
	cv2.destroyAllWindows()

# prosedur presensi wajah.
def faceAttendances(res):
	try:
		modeldir = './model/20170512-110547.pb'
		classifier_filename = './class/classifier.pkl'
		npy='./npy'
		modellive = './model/liveness.model'
		thepickle = './model/le.pickle'
		with tf.Graph().as_default():
		    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
		    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		    with sess.as_default():
		        pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
		        minsize = 20  # minimum size of face
		        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
		        factor = 0.709  # scale factor
		        margin = 44
		        frame_interval = 3
		        batch_size = 1000
		        image_size = 182
		        input_image_size = 160

		        src.facenet.load_model(modeldir)
		        model = load_model(modellive)

		        le = pickle.loads(open(thepickle, "rb").read())
		        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		        embedding_size = embeddings.get_shape()[1]

		        with open(classifier_filename, 'rb') as f:
		        	embeddingDataset, label = pickle.load(f)

		       	names = cariNama()

		        video_capture = cv2.VideoCapture(0)
		        # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
		        # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
		        c = 0
		        fps_list = []
		        tmp_time = time.time()
		        prevTime = 0
		        font = cv2.FONT_HERSHEY_PLAIN
		        attendances = []
		        thename = []
		        datenow = []
		        statusWaktus = []
		        sesii = []
		        waktus = []
		        mostName = ""
		        frequency = 2500
		        duration = 1000
		        j=0
		        while True:
		        	try:
		        		ret, frame = video_capture.read()
		        		delay = time.time() - tmp_time
		        		tmp_time = time.time()
		        		fps_list.append(delay)
		        		fps = len(fps_list) / np.sum(fps_list)
		        		if not ret:
		        			print("video has ended")
		        			break
		        		# Init
		        		now = datetime.now()
		        		dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
		        		datenows = now.strftime("%d-%m-%Y")
		        		hoursnow = now.strftime("%H:%M:%S")
		        		waktu, statusWaktu, sesis = treshHours(hoursnow)
		        		dirName = datenows
		        		frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
		        		minReso = 1
		        		small_frame = cv2.resize(frame, (0, 0), fx=1/minReso, fy=1/minReso)
		        		curTime = time.time()+1
		        		timeF = frame_interval
		        		find_results = []
		        		if frame.ndim == 2:
		        			frame = src.facenet.to_rgb(frame)
		        		rgb_small_frame = small_frame[:, :, ::-1]
		        		bounding_boxes, points = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)
		        		# print(points)
		        		nrof_faces = bounding_boxes.shape[0]
		        		cv2.putText(frame,("Press [ESC] to exit"),(10, 20),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("FPS: "+str("%.2f" % fps)),(10, 40),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("Student Presenced: "+str(len(attendances))),(10, 60),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("Attendance Frame: "+str(j)),(10, 80),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("Presenced: "+str(mostName)),(10, 100),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("Face Detected: "+str(nrof_faces)),(10, 120),font,1,STD_COLORS['black'],1)
		        		n = 20
		        		if nrof_faces > 0:
		        			try:
		        				det = bounding_boxes[:, 0:4]
		        				img_size = np.asarray(frame.shape)[0:2]
		        				cropped = []
		        				scaled = []
		        				scaled_reshape = []
		        				bb = np.zeros((nrof_faces,4), dtype=np.int32)
		        				for i in range(nrof_faces):
		        					emb_array = np.zeros((1, embedding_size))
		        					bb[i][0] = det[i][0] * minReso
		        					bb[i][1] = det[i][1] * minReso
		        					bb[i][2] = det[i][2] * minReso
		        					bb[i][3] = det[i][3] * minReso
		        					w = bb[i][2]-bb[i][0]
		        					h = bb[i][3]-bb[i][1]
		        					distance = (2*3.14 * 180)/(w+h*360)*1000
		        					distanceCM = math.floor(distance*2+distance*2)
		        					cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
		        					cropped[i] = src.facenet.flip(cropped[i], False)
		        					scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
		        					scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
		        					scaled[i] = src.facenet.prewhiten(scaled[i])
		        					scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
		        					feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
		        					emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
		        					dis = src.facenet.distance(embeddingDataset, emb_array[0,:])
		        					best_class_indices = np.argmin(dis)
		        					mindis = min(dis)
		        					face = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
		        					face = cv2.resize(face, (32, 32))
		        					face = face.astype("float") / 255.0
		        					face = img_to_array(face)
		        					face = np.expand_dims(face, axis=0)
		        					preds = model.predict(face)[0]
		        					j = np.argmax(preds)
		        					labels = le.classes_[j]
		        					if labels == "Asli":
		        						if mindis<0.9:
		        							ids = label[best_class_indices]
		        							id = names[ids]
		        							if distanceCM<45:
		        								color=STD_COLORS['green']
		        								warn = "Tetap pada posisi!"
		        							else:
		        								color=STD_COLORS['red']
		        								warn = "Tolong mendekat!"
	        								if id == 'Unknown':
	        									warn = "Tidak dikenal"
	        									color = STD_COLORS['white']
	        									cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
	        									cv2.putText(frame,warn,(bb[i][0], bb[i][1]-20),font,1,color,2)
	        								else:
	        									profile = getProfile(id)
	        									name = profile[2]
	        									sex = profile[4]
	        									status = profile[7]
	        									dist = mindis
	        									if name in attendances:
	        										color = STD_COLORS['white']
	        										warn = "Selamat, kamu hadir!"
        										cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
        										cv2.putText(frame,warn,(bb[i][0], bb[i][1]-20),font,1,color,2)
        										cv2.putText(frame, "Euc: "+str("%.2f" % dist), (bb[i][0], bb[i][3] + 80), font, 1, color, thickness=1, lineType=2)
        										cv2.putText(frame, "Nama: "+str(name), (bb[i][0], bb[i][3] + 20), font, 1, color, thickness=1, lineType=2)
        										cv2.putText(frame, "JK: "+str(sex), (bb[i][0], bb[i][3] + 40), font, 1, color, thickness=1, lineType=2)
        										cv2.putText(frame, "Status: "+str(status), (bb[i][0], bb[i][3] + 60), font, 1, color, thickness=1, lineType=2)
        										batas = 10
        										if distanceCM<=45:
        											if name not in attendances:
        												j+=1
        												thename.append(name)
        											if len(thename) == batas:
        												winsound.Beep(frequency, duration)
        												facess = Counter(thename)
        												mostFaces = facess.most_common(10)
        												nameMF = [i[0] for i in mostFaces]
        												mostName = nameMF[0]
        												if mostName not in attendances:
        													attendances.append(mostName)
        													datenow.append(hoursnow)
        													waktus.append(waktu)
        													statusWaktus.append(statusWaktu)
        													sesii.append(sesis)
        													thename=[]
        													j=0
		        						else:
		        							warn = "Tidak dikenal"
		        							color = STD_COLORS['white']
		        							cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
		        							cv2.putText(frame,warn,(bb[i][0], bb[i][1]-20),font,1,color,2)
		        					else:
		        						cv2.putText(frame, "Fake Face", (bb[i][0]+10, bb[i][1]-10), font, 1, STD_COLORS['red'], thickness=2, lineType=2)
		        						datenows = now.strftime("%d-%m-%Y")
		        						dirName = datenows
		        						if not os.path.exists("maybeFake/"+dirName):
		        							os.mkdir("maybeFake/"+dirName)
		        							print("Directory " , dirName ,  " Created ")
		        						else:
		        							DIR = "maybeFake/"+str(dirName)
		        							totalItems = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		        							cv2.imwrite("maybeFake/"+dirName+"/"+str(totalItems+1)+".jpg",frame)
		        							print("maybeFake/"+dirName+"/"+str(totalItems+1)+".jpg saved to maybeFake")
		        			except:
		        				pass
		        		cv2.imshow('Live Camera', frame)
		        		k = cv2.waitKey(33)
		        		if k==27:
		        			break
		        	except:
		        		pass
		            
		        video_capture.release()
		        cv2.destroyAllWindows()
		        presencedlist = [list(a) for a in zip(attendances,datenow,waktus,statusWaktus,sesii)]
		        now = datetime.now()
		        hours = now.strftime("%H-%M-%S")
		        df = pd.DataFrame(data=presencedlist,columns=['Nama','Waktu Presensi','Waktu','Status', 'Sesi'])
		        df.index += 1
		        df.style.set_properties(subset=['Nama','Waktu Presensi','Waktu','Status', 'Sesi'],**{'text-align':'center'})
		        if not os.path.exists("Presensi/"+dirName):
		        	os.mkdir("Presensi/"+dirName)
		        	print("Directory " , dirName ,  " Created ")
		        writer = pd.ExcelWriter("Presensi/"+str(dirName)+"/"+str(hours)+'.xlsx', engine='xlsxwriter')
		        df.to_excel(writer, sheet_name='Sheet1')
		        workbook = writer.book
		        worksheet = writer.sheets['Sheet1']
		        worksheet.set_column('B:B',18)
		        worksheet.set_column('C:C',18)
		        worksheet.set_column('D:D',18)
		        worksheet.set_column('E:E',18)
		        worksheet.set_column('F:F',18)
		        writer.save()
	except:
		pass

# prosedur uji spoof, akurasi (l2 distance), dan menampilkan chart setelah pengambilan video berakhir.
def detectFaces(cam,res):
	# try:
		modeldir = './model/20170512-110547.pb'
		classifier_filename = './class/classifier.pkl'
		npy='./npy'
		modellive = './model/liveness.model'
		thepickle = './model/le.pickle'

		with tf.Graph().as_default():
		    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		    with sess.as_default():
		    	#Init MTCNN
		        pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
		        minsize = 20  # minimum size of face
		        threshold = [0.8, 0.8, 0.8]  # three steps's threshold
		        factor = 0.709  # scale factor
		        margin = 44
		        frame_interval = 3
		        batch_size = 1000
		        image_size = 640
		        input_image_size = 160

		        # Load model FaceNet
		        src.facenet.load_model(modeldir)
		        # Load model deteksi wajah palsu
		        model = load_model(modellive)
		        le = pickle.loads(open(thepickle, "rb").read())

		        # Init
		        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		        embedding_size = embeddings.get_shape()[1]
		        # print(embedding_size)

		        with open(classifier_filename, 'rb') as f:
		        	embeddingDataset, label = pickle.load(f)

		        # Mengambil nama (ID) dari folder path dataset
		       	names = cariNama()

		       	# Eksekusi kamera
		        video_capture = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
		        # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
		        # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

		        # Init Variabel
		        c = 0
		        fps_list = []
		        tmp_time = time.time()
		        prevTime = 0
		        font = cv2.FONT_HERSHEY_PLAIN
		        theframe = []
		        theaccuracy = []
		        thename = []
		        thedistance = []
		        accurdist = []
		        frameCalc = 0
		        while True:
		        	# try:
		        		ret, frame = video_capture.read()
		        		delay = time.time() - tmp_time
		        		tmp_time = time.time()
		        		fps_list.append(delay)
		        		fps = len(fps_list) / np.sum(fps_list)
		        		now = datetime.now()
		        		datenows = now.strftime("%d-%m-%Y")
		        		dirName = datenows
		        		# untuk menghitung frame
		        		frameCalc+=1
		        		# Jika frame video berakhir, looping akan dibreak.
		        		if not ret:
		        			print("video has ended")
		        			break
		        		# Resize Frame 80%
		        		frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
		        		small_frame = cv2.resize(frame, (0, 0), fx=1/res, fy=1/res)
		        		rgb_small_frame = small_frame[:, :, ::-1]
		        		bounding_boxes, pointss = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)
		        		if frame.ndim == 2:
		        			frame = src.facenet.to_rgb(frame)
		        		nrof_faces = bounding_boxes.shape[0]
		        		cv2.putText(frame,("Press [ESC] to exit"),(10, 20),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("FPS: "+str("%.2f" % fps)),(10, 40),font,1,STD_COLORS['black'],1)
		        		cv2.putText(frame,("Face Detected: "+str(nrof_faces)),(10, 60),font,1,STD_COLORS['black'],1)
		        		if nrof_faces > 0:
		        			# try:
		        				# mengambil titik landmark
			        			lan = pointss[:, 0:2]
			        			# mengambil koordinat wajah (4 titik)
			        			det = bounding_boxes[:, 0:4]
			        			## cc di init seperti ini untuk mengubah tipe data floatnya lan (titik landmark), agar bisa dipakai pada opencv
			        			# cc = np.zeros((landmarks,2), dtype=np.int32)
			        			# bb juga sama seperti cc, tetapi untuk koordinat wajah.
			        			bb = np.zeros((nrof_faces,4), dtype=np.int32)
			        			img_size = np.asarray(frame.shape)[0:2]
			        			cropped = []
			        			scaled = []
			        			scaled_reshape = []
			        			cropped2 = []
			        			scaled2 = []
			        			scaled_reshape2 = []
			        			for banyakWajah in range(nrof_faces):
			        				emb_array = np.zeros((1, embedding_size))
			        				for h in range(4):
			        					bb[banyakWajah][h] = det[banyakWajah][h] * res
		        					w = bb[banyakWajah][2]-bb[banyakWajah][0]
		        					h = bb[banyakWajah][3]-bb[banyakWajah][1]
		        					distance = (2*3.14 * 180)/(w+h*360)*1000
		        					distanceCM = math.floor(distance*4)
		        					cropped.append(frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :])
		        					cropped[banyakWajah] = src.facenet.flip(cropped[banyakWajah], False)
		        					scaled.append(misc.imresize(cropped[banyakWajah], (image_size, image_size), interp='bilinear'))
		        					scaled[banyakWajah] = cv2.resize(scaled[banyakWajah], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
		        					scaled[banyakWajah] = src.facenet.prewhiten(scaled[banyakWajah])
		        					scaled_reshape.append(scaled[banyakWajah].reshape(-1,input_image_size,input_image_size,3))
		        					feed_dict = {images_placeholder: scaled_reshape[banyakWajah], phase_train_placeholder: False}
		        					emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
		        					dis = src.facenet.distance(embeddingDataset, emb_array)
		        					best_class_indices = np.argmin(dis)
		        					mindis = min(dis)
		        					face = frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :]
		        					face = cv2.resize(face, (32, 32))
		        					face = face.astype("float") / 255.0
		        					face = img_to_array(face)
		        					face = np.expand_dims(face, axis=0)
		        					# memprediksi bagian wajah yang sudah terpotong dengan model liveness yang sudah di train (real/fake)
		        					preds = model.predict(face)[0]
		        					j = np.argmax(preds)
		        					labels = le.classes_[j]
		        					labelss = "{}: {:.4f}".format(labels, preds[j])
		        					if labels == "Asli":
		        						if mindis<0.8:
		        							ids = label[best_class_indices]
		        							id = names[ids]
		        							profile = getProfile(id)
		        							name = profile[2]
		        							sex = profile[4]
		        							status = profile[7]
		        							colors = profile[8]
		        							dist = mindis
		        							color = STD_COLORS[colors]
		        							cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
		        							cv2.putText(frame, "E.Distance: "+str("%.2f" % dist), (bb[banyakWajah][0], bb[banyakWajah][3] + 80), font, 1, color, thickness=1, lineType=2)
		        							cv2.putText(frame, "Nama: "+str(name), (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
		        							cv2.putText(frame, "JK: "+str(sex), (bb[banyakWajah][0], bb[banyakWajah][3] + 40), font, 1, color, thickness=1, lineType=2)
		        							cv2.putText(frame, "Status: "+str(status), (bb[banyakWajah][0], bb[banyakWajah][3] + 60), font, 1, color, thickness=1, lineType=2)
		        							cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
		        							theframe.append(frameCalc)
		        							thename.append(name)
		        							theaccuracy.append(dist)
		        							thedistance.append(distanceCM)
		        							if len(theaccuracy) == 0 or len(thedistance)==0:
		        								print("Could not recognize the face")
		        							else:
		        								accurdist = [list(a) for a in zip(theaccuracy, thedistance)]
	        							else:
	        								name = 'Unknown'
	        								color = STD_COLORS['white']
	        								cv2.putText(frame,name, (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
	        								cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
	        								labels = "{}: {:.4f}".format(labels, preds[j])
	        								cv2.putText(frame, labels, (bb[banyakWajah][0]+10, bb[banyakWajah][1] - 30), font, 1, color, thickness=1, lineType=2)
	        								cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
        							else:
        								color = STD_COLORS['red']
        								cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
        								cv2.putText(frame, "Fake Face", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-60), font, 1, color , thickness=2, lineType=2)
        								cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
        								cv2.putText(frame, labelss, (bb[banyakWajah][0]+10, bb[banyakWajah][1] -30), font, 1, color, thickness=1, lineType=2)
		        			# except:
		        			# 	pass
		        		cv2.imshow('Live Camera', frame)
		        		# Jika pencet ESC akan exit
		        		k = cv2.waitKey(33)
		        		if k==27:
		        			break
		        	# except:
		        	# 	pass
		            
		        # jika tidak ada wajah yang dikenal terdeteksi maka akan muncul no known face detected
		        if len(thename) == 0:
		        	print("No known face detected!")
		        else:
		        	# Counter muka / mengelompokan nama yang sama dalam list.
		        	facess = Counter(thename)
		        	# mengambil 10 nama terbanyak
		        	mostFaces = facess.most_common(10)
		        	# mostFaces dipecah menjadi 2, yaitu nama dan frame
		        	nameMF = [i[0] for i in mostFaces]
		        	frameMF = [i[1] for i in mostFaces]
		        	# mengambil nama terbanyak yang keluar (indeks ke 0)
		        	mostName = nameMF[0]
		        if not len(accurdist) == 0:
		        	# sorting jarak dan akurasi, berdasarkan jarak sehingga tidak mengubah nilai akurasi
		        	accurdist = sorted(accurdist, key=itemgetter(1))
		        	# init
		        	length = len(accurdist)
		        	list2 = []
		        	index1 = []
		        	meanindex1 = []
		        	a = 0
		        	# melakukan perulangan untuk mendapatkan rata-rata dari akurasi yang memiliki jarak yang sama.
		        	# agar tampilan line chartnya baik.
		        	for i in range(length):
		        		index1.append(accurdist[i][0])
		        		if not a == accurdist[i][1]:
		        			index1 = mean(index1)
		        			meanindex1.append(index1)
		        			list2.append(accurdist[i][1])
		        			a = accurdist[i][1]
		        			index1 = []
		        	merged_list = [(meanindex1[i], list2[i]) for i in range(0, len(meanindex1))]
		        	newAcc = [i[0] for i in merged_list]
		        	newDist = [i[1] for i in merged_list]
		        video_capture.release()
		        cv2.destroyAllWindows()
		        if len(thename) == 0:
		        	messagebox.showinfo("info","No known face detected!")
		        else:
		        	MsgBox = messagebox.askquestion('Chart','most detected face is '+mostName+',\n do you want to see the graphic chart?', icon='warning')
		        	if MsgBox == 'yes':
		        		if not theframe == '' or theaccuracy == '' or nameMF == '' or frameMF == '' or newDist == '' or newAcc == '':
		        			chart(theframe,theaccuracy,nameMF,frameMF,newDist,newAcc)
	# except TypeError:
	# 	messagebox.showinfo("Infor","Eksekusi gagal!")


def saveVideo(cam,res):
	try:
		modeldir = './model/20170512-110547.pb'
		classifier_filename = './class/classifier.pkl'
		npy='./npy'
		modellive = './model/liveness.model'
		thepickle = './model/le.pickle'

		with tf.Graph().as_default():
		    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		    with sess.as_default():
		    	#Init MTCNN
		        pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
		        minsize = 20  # minimum size of face
		        threshold = [0.9, 0.9, 0.9]  # three steps's threshold
		        factor = 0.709  # scale factor
		        margin = 44
		        frame_interval = 3
		        batch_size = 1000
		        image_size = 182
		        input_image_size = 160

		        # Load model FaceNet
		        src.facenet.load_model(modeldir)
		        # Load model deteksi wajah palsu
		        model = load_model(modellive)
		        le = pickle.loads(open(thepickle, "rb").read())

		        # Init
		        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		        embedding_size = embeddings.get_shape()[1]
		        # print(embedding_size)


		        frame_number = 0
		        video_capture = cv2.VideoCapture(cam)
		        length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

		        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		        i = 0
		        videoName = "scanVideo/Video("+str(i)+")"+".mp4"
		        while True:
		        	if os.path.exists("scanVideo/Video("+str(i)+")"+".mp4") == True:
		        		i+=1
		        	else:
		        		print("scanVideo/Video("+str(i)+")"+".mp4")
		        		videoName = "scanVideo/Video("+str(i)+")"+".mp4"
		        		break
		        output_movie = cv2.VideoWriter(videoName, fourcc, 10, (width,height))

		        with open(classifier_filename, 'rb') as f:
		        	embeddingDataset, label = pickle.load(f)

		        # Mengambil nama (ID) dari folder path dataset
		       	names = cariNama()

		       	# Eksekusi kamera

		        # Init Variabel
		        fps_list = []
		        tmp_time = time.time()
		        prevTime = 0
		        font = cv2.FONT_HERSHEY_PLAIN
		        while True:
		            ret, frame = video_capture.read()
		            frame_number += 1
		            if not ret:
		                break
		            # Menghitung FPS
		            delay = time.time() - tmp_time
		            tmp_time = time.time()
		            fps_list.append(delay)
		            fps = len(fps_list) / np.sum(fps_list)

		            # Jika frame video berakhir, looping akan dibreak.
		            
		            
		            # Resize Frame 80%
		            # frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)

		            # Resize Frame untuk deteksi wajah (untuk dapat fps yang lebih kencang)
		            small_frame = cv2.resize(frame, (0, 0), fx=1/res, fy=1/res)
		            rgb_small_frame = small_frame[:, :, ::-1]

		            # deteksi wajah menggunakan MTCNN
		            bounding_boxes, pointss = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)

		            # Melihat framenya mempunyai berapa dimensional array
		            if frame.ndim == 2:
		                frame = src.facenet.to_rgb(frame)
		            
		            # .shape mereturnkan berapa banyak dimensional array yang ada (untuk melihat berapa banyak wajah yang terdeteksi)
		            nrof_faces = bounding_boxes.shape[0]
		            # untuk melihat banyak titik landmark (10)
		            # landmarks = pointss.shape[0]
		            cv2.putText(frame,("Press [ESC] to exit"),(10, 20),font,1,STD_COLORS['black'],1)
		            cv2.putText(frame,("FPS: "+str("%.2f" % fps)),(10, 40),font,1,STD_COLORS['black'],1)
		            cv2.putText(frame,("Face Detected: "+str(nrof_faces)),(10, 60),font,1,STD_COLORS['black'],1)
		            if nrof_faces > 0:
		            	try:
		            		# mengambil titik landmark
		            		lan = pointss[:, 0:2]
		            		# mengambil koordinat wajah (4 titik)
		            		det = bounding_boxes[:, 0:4]

		            		## cc di init seperti ini untuk mengubah tipe data floatnya lan (titik landmark), agar bisa dipakai pada opencv
		            		# cc = np.zeros((landmarks,2), dtype=np.int32)

		            		# bb juga sama seperti cc, tetapi untuk koordinat wajah.
		            		bb = np.zeros((nrof_faces,4), dtype=np.int32)
		            		# init
		            		img_size = np.asarray(frame.shape)[0:2]
		            		cropped = []
		            		scaled = []
		            		scaled_reshape = []
		            		
		            		# untuk multiface harus melakukan perulangan dengan panjang nilainya adalah banyak wajah yang terdeteksi.
		            		for banyakWajah in range(nrof_faces):

		            			# mengambil nilai pengukuran wajah real time
		            			emb_array = np.zeros((1, embedding_size))

		            			# 4 titik pada wajah, i = wajah ke berapa, dan h = 4 titik koordinat wajah
		            			for h in range(4):
		            				bb[banyakWajah][h] = det[banyakWajah][h] * res

		            			# w, h untuk mendapatkan width dan height: untuk menghitung jarak kamera degan muka
		            			w = bb[banyakWajah][2]-bb[banyakWajah][0]
		            			h = bb[banyakWajah][3]-bb[banyakWajah][1]

		            			# rumus untuk menghitung jarak, dapat dari internet dan tidak ada dasarnya dapat dari mana ini fungsi.
		            			distance = (2*3.14 * 180)/(w+h*360)*1000
		            			distanceCM = math.floor(distance*4)

		            			# untuk tiap wajah yang dideteksi pada frame, akan diambil bagian wajahnya saja untuk dibandingkan pengukurannya dengan
		            			# pengukuran dengan dataset.
		            			cropped.append(frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :])
		            			cropped[banyakWajah] = src.facenet.flip(cropped[banyakWajah], False)
		            			scaled.append(misc.imresize(cropped[banyakWajah], (image_size, image_size), interp='bilinear'))
		            			scaled[banyakWajah] = cv2.resize(scaled[banyakWajah], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
		            			scaled[banyakWajah] = src.facenet.prewhiten(scaled[banyakWajah])
		            			scaled_reshape.append(scaled[banyakWajah].reshape(-1,input_image_size,input_image_size,3))
		            			feed_dict = {images_placeholder: scaled_reshape[banyakWajah], phase_train_placeholder: False}
		            			emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

		            			# Menghitung jarak euclidean wajah pada dataset dan wajah yang terdeteksi pada frame, dan mengambil
		            			dis = src.facenet.distance(embeddingDataset, emb_array)
		            			# print("dis: "+str(dis))
		            			testt = np.linalg.norm(embeddingDataset - emb_array[0,:], axis=1)
		            			# print("test: "+str(testt))

		            			# test2 = np.sqrt(np.sum(np.square(np.subtract(embeddingDataset, emb_array[0,:]))))
		            			# print(test2)
		            			best_class_indices = np.argmin(dis)
		            			# 
		            			mindis = min(dis)
		            			
		            			# memotong bagian wajah yang terdeteksi pada frame untuk dilihat wajah asli atau palsu.
		            			face = frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :]
		            			face = cv2.resize(face, (32, 32))
		            			face = face.astype("float") / 255.0
		            			face = img_to_array(face)
		            			face = np.expand_dims(face, axis=0)

		            			# memprediksi bagian wajah yang sudah terpotong dengan model liveness yang sudah di train (real/fake)
		            			preds = model.predict(face)[0]
		            			j = np.argmax(preds)
		            			labels = le.classes_[j]

		            			## Titik landmark, j = 10 titik landmark (x,y), i = total wajah.
		            			# for panjangLandmark in range(landmarks):
		            			# 	cc[panjangLandmark][banyakWajah] = lan[panjangLandmark][banyakWajah] * res
		            			# for n in range(5):
		            			# 	cv2.circle(frame, (cc[n][banyakWajah],cc[n+5][banyakWajah]), 4, (255, 0, 0), -1)
		            			labelss = "{}: {:.4f}".format(labels, preds[j])
		            			if labels == "Asli":
		            				
		            				# if preds[j] > 0.9:
		            				if mindis<0.9:
			            				ids = label[best_class_indices]
			            				id = names[ids]
			            				profile = getProfile(id)
			            				name = profile[2]
			            				sex = profile[4]
			            				status = profile[7]
			            				colors = profile[8]
			            				dist = mindis
			            				color = STD_COLORS[colors]
			            				
			            				
			            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "E.Distance: "+str("%.2f" % dist), (bb[banyakWajah][0], bb[banyakWajah][3] + 80), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Nama: "+str(name), (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "JK: "+str(sex), (bb[banyakWajah][0], bb[banyakWajah][3] + 40), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Status: "+str(status), (bb[banyakWajah][0], bb[banyakWajah][3] + 60), font, 1, color, thickness=1, lineType=2)
			            				cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
			            			else:
			            				name = 'Unknown'
			            				color = STD_COLORS['white']
			            				cv2.putText(frame,name, (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
			            				cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
			            				labels = "{}: {:.4f}".format(labels, preds[j])
			            				cv2.putText(frame, labels, (bb[banyakWajah][0]+10, bb[banyakWajah][1] - 30), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
				            			
			            			# else:
			            			# 	cv2.putText(frame, "Fake Face", (bb[i][0]+10, bb[i][1]-10), font, 1, STD_COLORS['red'], thickness=2, lineType=2)
		            			else:
		            				color = STD_COLORS['red']
		            				cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
		            				cv2.putText(frame, "Fake Face", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-60), font, 1, color , thickness=2, lineType=2)
		            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
	            				cv2.putText(frame, labelss, (bb[banyakWajah][0]+10, bb[banyakWajah][1] -30), font, 1, color, thickness=1, lineType=2)
		            	except:
		            		pass
		            print("Writing frame {} / {}".format(frame_number, length))
		            cv2.waitKey(1)
		            # cv2.imshow('Capturing Faces', frame)
		            output_movie.write(frame)
		            
		        video_capture.release()
		        cv2.destroyAllWindows()
		        messagebox.showinfo("info","Detect Uploaded Video Saved!")
	except TypeError:
		messagebox.showinfo("Infor","Eksekusi gagal!")


def gatherLiveness(cam,res):
	try:
		modeldir = './model/20170512-110547.pb'
		classifier_filename = './class/classifier.pkl'
		npy='./npy'
		modellive = './model/liveness.model'
		thepickle = './model/le.pickle'

		with tf.Graph().as_default():
		    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		    with sess.as_default():
		    	#Init MTCNN
		        pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
		        minsize = 20  # minimum size of face
		        threshold = [0.9, 0.9, 0.9]  # three steps's threshold
		        factor = 0.709  # scale factor
		        margin = 44
		        frame_interval = 3
		        batch_size = 1000
		        image_size = 182
		        input_image_size = 160

		        # Load model FaceNet
		        src.facenet.load_model(modeldir)
		        # Load model deteksi wajah palsu
		        model = load_model(modellive)
		        le = pickle.loads(open(thepickle, "rb").read())

		        # Init
		        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		        embedding_size = embeddings.get_shape()[1]
		        # print(embedding_size)

		        with open(classifier_filename, 'rb') as f:
		        	embeddingDataset, label = pickle.load(f)

		        # Mengambil nama (ID) dari folder path dataset
		       	names = cariNama()

		       	# Eksekusi kamera
		        video_capture = cv2.VideoCapture(cam)
		        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
		        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

		        # Init Variabel
		        c = 0
		        fps_list = []
		        tmp_time = time.time()
		        prevTime = 0
		        font = cv2.FONT_HERSHEY_PLAIN
		        theframe = []
		        theaccuracy = []
		        thename = []
		        thedistance = []
		        accurdist = []
		        frameCalc = 0
		        while True:
		            ret, frame = video_capture.read()
		            # Menghitung FPS
		            delay = time.time() - tmp_time
		            tmp_time = time.time()
		            fps_list.append(delay)
		            fps = len(fps_list) / np.sum(fps_list)
		            now = datetime.now()
		            datenows = now.strftime("%d-%m-%Y")
		            dirName = datenows
		            # untuk menghitung frame
		            frameCalc+=1

		            # Jika frame video berakhir, looping akan dibreak.
		            if not ret:
		                print("video has ended")
		                break
		            
		            # Resize Frame 80%
		            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)

		            # Resize Frame untuk deteksi wajah (untuk dapat fps yang lebih kencang)
		            small_frame = cv2.resize(frame, (0, 0), fx=1/res, fy=1/res)
		            rgb_small_frame = small_frame[:, :, ::-1]

		            # deteksi wajah menggunakan MTCNN
		            bounding_boxes, pointss = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)

		            # Melihat framenya mempunyai berapa dimensional array
		            if frame.ndim == 2:
		                frame = src.facenet.to_rgb(frame)
		            
		            # .shape mereturnkan berapa banyak dimensional array yang ada (untuk melihat berapa banyak wajah yang terdeteksi)
		            nrof_faces = bounding_boxes.shape[0]
		            # untuk melihat banyak titik landmark (10)
		            # landmarks = pointss.shape[0]
		            cv2.putText(frame,("Press [ESC] to exit"),(10, 20),font,1,STD_COLORS['black'],1)
		            cv2.putText(frame,("FPS: "+str("%.2f" % fps)),(10, 40),font,1,STD_COLORS['black'],1)
		            cv2.putText(frame,("Face Detected: "+str(nrof_faces)),(10, 60),font,1,STD_COLORS['black'],1)
		            if nrof_faces > 0:
		            	try:
		            		# mengambil titik landmark
		            		lan = pointss[:, 0:2]
		            		# mengambil koordinat wajah (4 titik)
		            		det = bounding_boxes[:, 0:4]

		            		## cc di init seperti ini untuk mengubah tipe data floatnya lan (titik landmark), agar bisa dipakai pada opencv
		            		# cc = np.zeros((landmarks,2), dtype=np.int32)

		            		# bb juga sama seperti cc, tetapi untuk koordinat wajah.
		            		bb = np.zeros((nrof_faces,4), dtype=np.int32)
		            		# init
		            		img_size = np.asarray(frame.shape)[0:2]
		            		cropped = []
		            		scaled = []
		            		scaled_reshape = []
		            		
		            		# untuk multiface harus melakukan perulangan dengan panjang nilainya adalah banyak wajah yang terdeteksi.
		            		for banyakWajah in range(nrof_faces):

		            			# mengambil nilai pengukuran wajah real time
		            			emb_array = np.zeros((1, embedding_size))

		            			# 4 titik pada wajah, i = wajah ke berapa, dan h = 4 titik koordinat wajah
		            			for h in range(4):
		            				bb[banyakWajah][h] = det[banyakWajah][h] * res

		            			# w, h untuk mendapatkan width dan height: untuk menghitung jarak kamera degan muka
		            			w = bb[banyakWajah][2]-bb[banyakWajah][0]
		            			h = bb[banyakWajah][3]-bb[banyakWajah][1]

		            			# rumus untuk menghitung jarak, dapat dari internet dan tidak ada dasarnya dapat dari mana ini fungsi.
		            			distance = (2*3.14 * 180)/(w+h*360)*1000
		            			distanceCM = math.floor(distance*4)

		            			# untuk tiap wajah yang dideteksi pada frame, akan diambil bagian wajahnya saja untuk dibandingkan pengukurannya dengan
		            			# pengukuran dengan dataset.
		            			cropped.append(frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :])
		            			cropped[banyakWajah] = src.facenet.flip(cropped[banyakWajah], False)
		            			scaled.append(misc.imresize(cropped[banyakWajah], (image_size, image_size), interp='bilinear'))
		            			scaled[banyakWajah] = cv2.resize(scaled[banyakWajah], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
		            			scaled[banyakWajah] = src.facenet.prewhiten(scaled[banyakWajah])
		            			scaled_reshape.append(scaled[banyakWajah].reshape(-1,input_image_size,input_image_size,3))
		            			feed_dict = {images_placeholder: scaled_reshape[banyakWajah], phase_train_placeholder: False}
		            			emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

		            			# Menghitung jarak euclidean wajah pada dataset dan wajah yang terdeteksi pada frame, dan mengambil
		            			dis = src.facenet.distance(embeddingDataset, emb_array)
		            			# print("dis: "+str(dis))
		            			testt = np.linalg.norm(embeddingDataset - emb_array[0,:], axis=1)
		            			# print("test: "+str(testt))

		            			# test2 = np.sqrt(np.sum(np.square(np.subtract(embeddingDataset, emb_array[0,:]))))
		            			# print(test2)
		            			best_class_indices = np.argmin(dis)
		            			# 
		            			mindis = min(dis)
		            			
		            			# memotong bagian wajah yang terdeteksi pada frame untuk dilihat wajah asli atau palsu.
		            			face = frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :]
		            			face = cv2.resize(face, (32, 32))
		            			face = face.astype("float") / 255.0
		            			face = img_to_array(face)
		            			face = np.expand_dims(face, axis=0)

		            			# memprediksi bagian wajah yang sudah terpotong dengan model liveness yang sudah di train (real/fake)
		            			preds = model.predict(face)[0]
		            			j = np.argmax(preds)
		            			labels = le.classes_[j]

		            			## Titik landmark, j = 10 titik landmark (x,y), i = total wajah.
		            			# for panjangLandmark in range(landmarks):
		            			# 	cc[panjangLandmark][banyakWajah] = lan[panjangLandmark][banyakWajah] * res
		            			# for n in range(5):
		            			# 	cv2.circle(frame, (cc[n][banyakWajah],cc[n+5][banyakWajah]), 4, (255, 0, 0), -1)
		            			labelss = "{}: {:.4f}".format(labels, preds[j])
		            			if labels == "Asli":
		            				if not os.path.exists("spoof/maybeReal/"+dirName):
		            					os.mkdir("spoof/maybeReal/"+dirName)
		            					print("Directory " , dirName ,  " Created ")
		            				else:
		            					DIR = "spoof/maybeReal/"+str(dirName)+"/"
		            					filenamessss = str(DIR)+str(randomString(8))+".jpg"
		            					cv2.imwrite(filenamessss,frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :])
		            					print(filenamessss+"saved to spoof/maybeReal")
		            				if mindis<0.8:
			            				ids = label[best_class_indices]
			            				id = names[ids]
			            				profile = getProfile(id)
			            				name = profile[2]
			            				sex = profile[4]
			            				status = profile[7]
			            				colors = profile[8]
			            				dist = mindis
			            				color = STD_COLORS[colors]
			            				
			            				
			            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "E.Distance: "+str("%.2f" % dist), (bb[banyakWajah][0], bb[banyakWajah][3] + 80), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Nama: "+str(name), (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "JK: "+str(sex), (bb[banyakWajah][0], bb[banyakWajah][3] + 40), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Status: "+str(status), (bb[banyakWajah][0], bb[banyakWajah][3] + 60), font, 1, color, thickness=1, lineType=2)
			            				# cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
			            				theframe.append(frameCalc)
			            				thename.append(name)
			            				theaccuracy.append(dist)
			            				thedistance.append(distanceCM)
			            				if len(theaccuracy) == 0 or len(thedistance)==0:
			            					print("Could not recognize the face")
			            				else:
			            					accurdist = [list(a) for a in zip(theaccuracy, thedistance)]
			            			else:
			            				name = 'Unknown'
			            				color = STD_COLORS['white']
			            				cv2.putText(frame,name, (bb[banyakWajah][0], bb[banyakWajah][3] + 20), font, 1, color, thickness=1, lineType=2)
			            				# cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
			            				labels = "{}: {:.4f}".format(labels, preds[j])
			            				cv2.putText(frame, labels, (bb[banyakWajah][0]+10, bb[banyakWajah][1] - 30), font, 1, color, thickness=1, lineType=2)
			            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
			            			
			            			# else:
			            			# 	cv2.putText(frame, "Fake Face", (bb[i][0]+10, bb[i][1]-10), font, 1, STD_COLORS['red'], thickness=2, lineType=2)
		            			else:
		            				color = STD_COLORS['red']
		            				# cv2.rectangle(frame, (bb[banyakWajah][0], bb[banyakWajah][1]), (bb[banyakWajah][2], bb[banyakWajah][3]), color, 2)
		            				cv2.putText(frame, "Fake Face", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-60), font, 1, color , thickness=2, lineType=2)
		            				cv2.putText(frame, "Jarak: "+str(distanceCM)+" cm", (bb[banyakWajah][0]+10, bb[banyakWajah][1]-10), font, 1, color, thickness=1, lineType=2)
		            				
		            				if not os.path.exists("spoof/maybeFake/"+dirName):
		            					os.mkdir("spoof/maybeFake/"+dirName)
		            					print("Directory " , dirName ,  " Created ")
		            				else:
		            					DIR = "spoof/maybeFake/"+str(dirName)+"/"
		            					totalItems = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		            					filenamessss = str(DIR)+str(randomString(8))+".jpg"
		            					cv2.imwrite(filenamessss,frame[bb[banyakWajah][1]:bb[banyakWajah][3], bb[banyakWajah][0]:bb[banyakWajah][2], :])
		            					print(filenamessss+" saved to spoof/maybeFake")
	            				cv2.putText(frame, labelss, (bb[banyakWajah][0]+10, bb[banyakWajah][1] -30), font, 1, color, thickness=1, lineType=2)
		            	except:
		            		pass
		            # Jika pencet ESC akan exit
		            k = cv2.waitKey(33)
		            if k==27:
		                break
		        # jika tidak ada wajah yang dikenal terdeteksi maka akan muncul no known face detected
		        if len(thename) == 0:
		        	print("No known face detected!")
		        else:
		        	# Counter muka / mengelompokan nama yang sama dalam list.
		        	facess = Counter(thename)
		        	# mengambil 10 nama terbanyak
		        	mostFaces = facess.most_common(10)
		        	# mostFaces dipecah menjadi 2, yaitu nama dan frame
		        	nameMF = [i[0] for i in mostFaces]
		        	frameMF = [i[1] for i in mostFaces]
		        	# mengambil nama terbanyak yang keluar (indeks ke 0)
		        	mostName = nameMF[0]
		        if not len(accurdist) == 0:
		        	# sorting jarak dan akurasi, berdasarkan jarak sehingga tidak mengubah nilai akurasi
		        	accurdist = sorted(accurdist, key=itemgetter(1))
		        	# init
		        	length = len(accurdist)
		        	list2 = []
		        	index1 = []
		        	meanindex1 = []
		        	a = 0
		        	# melakukan perulangan untuk mendapatkan rata-rata dari akurasi yang memiliki jarak yang sama.
		        	# agar tampilan line chartnya baik.
		        	for i in range(length):
		        		index1.append(accurdist[i][0])
		        		if not a == accurdist[i][1]:
		        			index1 = mean(index1)
		        			meanindex1.append(index1)
		        			list2.append(accurdist[i][1])
		        			a = accurdist[i][1]
		        			index1 = []
		        	merged_list = [(meanindex1[i], list2[i]) for i in range(0, len(meanindex1))]
		        	newAcc = [i[0] for i in merged_list]
		        	newDist = [i[1] for i in merged_list]
		        video_capture.release()
		        cv2.destroyAllWindows()
		        if len(thename) == 0:
		        	messagebox.showinfo("info","No known face detected!")
		        else:
		        	MsgBox = messagebox.askquestion('Chart','most detected face is '+mostName+',\n do you want to see the graphic chart?', icon='warning')
		        	if MsgBox == 'yes':
		        		if not theframe == '' or theaccuracy == '' or nameMF == '' or frameMF == '' or newDist == '' or newAcc == '':
		        			chart(theframe,theaccuracy,nameMF,frameMF,newDist,newAcc)
	except TypeError:
		messagebox.showinfo("Infor","Eksekusi gagal!")


# prosedur menambahkan foto menggunakan live cam.
def liveRecordss(id):
	font= cv2.FONT_HERSHEY_PLAIN
	if id == '':
		messagebox.showinfo("Info", "ID harap diisi!")
	else:
		conn = sqlite3.connect('Facebase.db')			
		c = conn.cursor()
		query = "SELECT * From Peoples JOIN Positions USING('PositionID') WHERE PeopleID="+id
		c.execute(query)
		record3 = c.fetchall()
		print_records4=''
		for record3 in record3:
			print_records4 += str(record3[2])+", "+str(record3[7])
			existrecord1 = 1
		if print_records4 == "":
			messagebox.showinfo("Info", "ID "+id+" belum ada dalam database")
		else:
			MsgBox1 = messagebox.askquestion('Add Photo',''+str(record3[2])+' will be updated', icon='warning')
			if MsgBox1 == 'yes':
				npy='./npy'
				q=-200+140
				w=-80+40
				e=210-140
				r=160-140
				sampleStart = 0
				sampleEnd = 15
				font= cv2.FONT_HERSHEY_PLAIN
				with tf.Graph().as_default():
				    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
				    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
				    with sess.as_default():
				    	DIR = "./images/1train_img/"+str(id)
				    	pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
				    	minsize = 20  # minimum size of face
				    	threshold = [0.6, 0.7, 0.7]  # three steps's threshold
				    	factor = 0.709  # scale factor
				    	minReso = 8
				    	video_capture = cv2.VideoCapture(0)
				    	video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
				    	video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
				    	while True:
				    		ret, frame = video_capture.read()
				    		frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
				    		small_frame = cv2.resize(frame, (0, 0), fx=1/minReso, fy=1/minReso)
				    		rgb_small_frame = small_frame[:, :, ::-1]
				    		bounding_boxes, _ = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)
				    		nrof_faces = bounding_boxes.shape[0]
				    		bb = np.zeros((nrof_faces,4), dtype=np.int32)
				    		if nrof_faces > 0:
				    			det = bounding_boxes[:, 0:4]
				    			for i in range(nrof_faces):
				    				bb[i][0] = (det[i][0] * minReso)+q
				    				bb[i][1] = (det[i][1] * minReso)+w
				    				bb[i][2] = (det[i][2] * minReso)+e
				    				bb[i][3] = (det[i][3] * minReso)+r
				    				cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0,0,0), 2)
				    				cropped_temp = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
		    				cv2.putText(frame,("Press 'Q' to snap your faces!"),(10, 20),font,2,(0,255,0),2)
		    				cv2.putText(frame,("User ID: "+str(id)),(10, 40),font,1,(0,0,0),1)
		    				cv2.imshow('Capturing Faces', frame)
		    				if cv2.waitKey(1) & 0xFF == ord('q'):
		    					totalItems = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		    					cv2.imwrite(str(DIR)+"/"+str(totalItems+1)+".jpg",cropped_temp)
		    					break
    					video_capture.release()
    					cv2.destroyAllWindows()
    					messagebox.showinfo("Info", "Face saved into "+str(DIR)+"/"+str(totalItems+1)+".jpg")

# prosedur membuat id baru.
def create(names, positions, genders, ages):
	npy='./npy'
	conn = sqlite3.connect('./Facebase.db')
	c = conn.cursor()
	query = "INSERT INTO Peoples(Name, PositionID, Gender, Age) VALUES ('"+str(names)+"','"+str(positions)+"','"+str(genders)+"',"+str(ages)+")"
	c.execute(query)
	PeopleID = c.lastrowid
	conn.commit()
	conn.close()
	font= cv2.FONT_HERSHEY_PLAIN
	with tf.Graph().as_default():
	    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	    with sess.as_default():
	        pnet, rnet, onet = src.detect_face.create_mtcnn(sess, npy)
	        minsize = 20  # minimum size of face
	        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
	        factor = 0.709  # scale factor
	        minReso = 8
	        video_capture = cv2.VideoCapture(0)
	        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
	        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
	        q=-200+140
	        w=-80+40
	        e=210-140
	        r=160-140
	        while True:
	        	ret, frame = video_capture.read()
	        	frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
	        	small_frame = cv2.resize(frame, (0, 0), fx=1/minReso, fy=1/minReso)
	        	rgb_small_frame = small_frame[:, :, ::-1]
	        	bounding_boxes, _ = src.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)
	        	nrof_faces = bounding_boxes.shape[0]
	        	bb = np.zeros((nrof_faces,4), dtype=np.int32)
	        	if nrof_faces > 0:
	        		det = bounding_boxes[:, 0:4]
	        		for i in range(nrof_faces):
	        			bb[i][0] = (det[i][0] * minReso)+q
	        			bb[i][1] = (det[i][1] * minReso)+w
	        			bb[i][2] = (det[i][2] * minReso)+e
	        			bb[i][3] = (det[i][3] * minReso)+r
	        			cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0,0,0), 2)
	        			cropped_temp = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
	        			if not os.path.exists("./images/1train_img/"+str(PeopleID)):
	        				os.mkdir("./images/1train_img/"+str(PeopleID))
	        				print("Directory " , str(PeopleID) ,  " Created ")			
		        cv2.putText(frame,("Press 'Q' to snap your faces!"),(10, 20),font,2,(0,255,0),2)
		        cv2.putText(frame,("User ID: "+str(PeopleID)),(10, 40),font,1,(0,0,0),1)
		        cv2.imshow('Capturing Faces', frame)
		        if cv2.waitKey(1) & 0xFF == ord('q'):
		        	cv2.imwrite("./images/1train_img/"+str(PeopleID)+"/1.jpg",cropped_temp)
		        	break
	        video_capture.release()
	        cv2.destroyAllWindows()
	        messagebox.showinfo("info","ID "+str(PeopleID)+" Created!")