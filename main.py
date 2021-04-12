#================ Library & File Pendukung ===================
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from src.align_face import align
from src.preprocess import preprocesses
from src.classifier import training
from tkinter import filedialog
import sqlite3
import os
import src.liveCam
import shutil
# ================== 	END Library & File Pendukung 	=======================
# ================== 	Init GUI						=======================
root = Tk()
root.resizable(0,0)
root.title('Face Recognition -  Face Attendances - Deep Learning - Wailan Tirajoh')
#========================== RESET FUNCTION ================================
'''
	RESET function dipakai untuk mengatasi GUI yang bertumpuk saat dipanggil,
	akan dilihat jika nilai dari tampilan yang dipencet, jika nilainya 1 maka grid akan dimatikan
'''
def reset():
	if detectorExist == 1:
		detectorLabel.grid_forget()
	if trainerExist == 1:
		labelTrainer.grid_forget()
	if creatorExist == 1:
		label1.grid_forget()
	if databaseAllExist == 1:
		databases.grid_forget()
	if loginExist == 1:
		logins.grid_forget()

def resetDb():
	if editExist == 1:
			editForm.grid_forget()
	if deleteExist == 1:
			deleteForm.grid_forget()
	if imageexist == 1:
			imageForm.grid_forget()

def resetDc():
	if editPositionsssFormExist == 1:
		editPositionsssForm.grid_forget()
	if addPositionssLabelFrameExist == 1:
		addPositionssLabelFrame.grid_forget()

def resetInsideDb():
	if databasePeopleExist == 1:
		databasePeopleShow.grid_forget()
	if databasePositionsssExist == 1:
		databasePositionsshow.grid_forget()
#======================== END OF RESET FUNCTION ============================
#========================	MAIN FUNCTION 	================================
''' 
	train image dipakai untuk classifier / pengenalan wajah, dilatih dan akan mengembalikan nilai 128 dimensional array per wajah
	nilai ini yang nantinya akan dipakai sebagai anchor untuk membandingkan wajah dalam dataset dengan wajah saat recording (live)
	data dilatih dengan menggunakan pre-trained model dari FaceNet.
'''
def train_image():
	datadir = './images/2align_face'
	modeldir = './model/20170512-110547.pb'
	classifier_filename = './class/classifier.pkl'
	print ("Training Start")
	obj=training(datadir,modeldir,classifier_filename)
	get_file=obj.main_train()
	print('Saved classifier model to file "%s"' % get_file)
	messagebox.showinfo("info","Train Successfully!")
'''
	data_preprcess dipakai untuk memotong wajah dan meresize wajah agar lebih rapih lagi agar classifier wajah dapat dilakukan
	dengan maksimal
'''
def data_preprocess():
	input_datadir = './images/2align_face'
	output_datadir = './images/3pre_img'
	obj=preprocesses(input_datadir,output_datadir)
	nrof_images_total,nrof_successfully_aligned=obj.collect_data()
	print('Total number of images: %d' % nrof_images_total)
	print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
	messagebox.showinfo("info","Preprocess Successfully!")
'''
	align_face dipakai untuk memperbaiki posisi wajah yang miring pada saat pengembalian gambar
'''
def align_face():
	input_datadir = './images/1train_img'
	output_datadir = './images/2align_face'
	obj=align(input_datadir,output_datadir)
	# obj.collect_data()
	nrof_images_total,nrof_successfully_aligned=obj.collect_data()
	print('Total number of images: %d' % nrof_images_total)
	print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
	messagebox.showinfo("info","Align Face Successfully!")
'''
	addPhotoFunction untuk menambahkan wajah pada dataset jika dirasa 1 wajah belum cukup.
'''
def addPhotoFunction():
	global addPhotoExist, labelAddPhoto
	if addPhotoExist == 1:
		labelAddPhoto.grid_forget()

	def liveRecord():
		src.liveCam.liveRecordss(id4.get())

	labelAddPhoto = LabelFrame(labelTrainer, text = "Add Photo ")
	labelAddPhoto.grid(row=5, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)

	id4 = Entry(labelAddPhoto, width=20)
	id4.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
	id_label = Label(labelAddPhoto, text="Masukkan ID")
	id_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5,sticky=W)

	liveRecords = Button(labelAddPhoto,text='Submit', command = liveRecord)
	liveRecords.grid(row=1, column=4, columnspan=2, padx=5,sticky=W, ipadx=10)

	addPhotoExist = 1

# =============================================	END OF MAIN FUNCTION =======================================================================
# ============================================= DETECTOR ===================================================================================
'''
	Detector adalah sekumpulan prosedur yang dibuat untuk hal pengujian, rekam, upload video untuk diuji, dan yang utama disini adalah
	presensi wajah
'''

'''
	Minres = Minimal Resolution, semakin tinggi min resolution yang dipakai, FPS akan semakin membaik tetapi
	wajah yang kecil akan sulit terdeteksi.
			 Min res yang kecil (paling kecil 1), akan menurunkan FPS, tetapi wajah yang kecil dapat terdeteksi.
'''
detectorExist = 0
def detector():
	global detectorLabel, detectorExist
	reset()
	def LiveCams():
		cam = 0
		minres = 1
		src.liveCam.detectFaces(cam,minres)
	def Records():
		src.liveCam.recordCam()
	def DetectRecords():
		folder_path = 'record/'
		dirList = os.listdir(folder_path)
		a = len(dirList)-1
		cam = 'record/video'+str(a)+'.mp4'
		minres = 1
		src.liveCam.detectFaces(cam,minres)
	def UploadFile():
		filename = filedialog.askopenfilename(initialdir = "/", title = "Select A File", filetype = (("mp4","*.mp4"),("All Files","*.*")))
		if not filename == '':
			cam = filename
			minres = 1
			src.liveCam.detectFaces(cam,minres)
	def UploadSave():
		filename = filedialog.askopenfilename(initialdir = "/", title = "Select A File", filetype = (("mp4","*.mp4"),("All Files","*.*")))
		if not filename == '':
			cam = filename
			minres = 1
			src.liveCam.saveVideo(cam,minres)
	def Attendance():
		minres = 1
		src.liveCam.faceAttendances(minres)
	def gatherLiveness():
		filename = filedialog.askopenfilename(initialdir = "/", title = "Select A File", filetype = (("mp4","*.mp4"),("All Files","*.*")))
		if not filename == '':
			cam = filename
			minres = 8
			src.liveCam.gatherLiveness(cam,minres)
		
		
# ================================================== DETECTOR GUI ===================================================================================
	detectorExist = 1

	detectorLabel = LabelFrame(labelRootChild, text = "Detectors")
	detectorLabel.grid(row=0, column=1, columnspan=7, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)

	LiveCam = Button(detectorLabel, text="LiveCam", command = LiveCams)
	LiveCam.config(width = 13)
	LiveCam.grid(row=0, column=1, columnspan=2, pady=10, padx=10)

	Record = Button(detectorLabel, text="Record", command = Records)
	Record.config(width = 13)
	Record.grid(row=0, column=3, columnspan=2, pady=10, padx=10)

	DetectRecord = Button(detectorLabel, text="Detect Record", command = DetectRecords)
	DetectRecord.grid(row=0, column=5, columnspan=2, pady=10, padx=10)
	DetectRecord.config(width = 13)

	UploadDetect = Button(detectorLabel, text="Upload", command = UploadFile)
	UploadDetect.grid(row=0, column=7, columnspan=2, pady=10, padx=10)
	UploadDetect.config(width = 13)

	UploadSave = Button(detectorLabel, text="UploadSave", command = UploadSave)
	UploadSave.grid(row=0, column=9, columnspan=2, pady=10, padx=10)
	UploadSave.config(width = 13)

	Attendancess = Button(detectorLabel, text="Attendance", command = Attendance)
	Attendancess.grid(row=0, column=11, columnspan=2, pady=10, padx=10)
	Attendancess.config(width = 13)

	gatherLivenessS = Button(detectorLabel, text="Gather Liveness", command = gatherLiveness)
	gatherLivenessS.grid(row=1, column=1, columnspan=2, pady=10, padx=10)
	gatherLivenessS.config(width = 13)

# ==================================================== TRAINER =============================================================================

addPhotoExist = 0
trainerExist = 0

def trainer():
	global trainerExist, labelTrainer
	reset()

	# ======================================================== TRAINER GUI =================================================================

	labelTrainer = LabelFrame(labelRootChild, text = "Data Trainer")
	labelTrainer.grid(row=1, column=1, columnspan=7, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)

	btnAlign = Button(labelTrainer, text="Align Face", command = align_face)
	btnAlign.grid(row=0, column=1, ipadx=10, padx=10, sticky=W)

	btnTrainImages = Button(labelTrainer, text="Preprocess", command = data_preprocess)
	btnTrainImages.grid(row=0, column=2, ipadx=10, padx=10, sticky=W)

	btnNormalized = Button(labelTrainer, text="Train Images", command = train_image)
	btnNormalized.grid(row=0, column=3, ipadx=10, padx=10, sticky=W)

	btnaddPhotoFrame = Button(labelTrainer, text="Add Photo's", command = addPhotoFunction)
	btnaddPhotoFrame.grid(row=0, column=4, ipadx=10, padx=10, sticky=W)

	
	trainerExist = 1

# ================================================ CREATOR =================================================================================

creatorExist = 0
def creator():
	global creatorExist, label1
	reset()
	creatorExist = 1
	label1 = LabelFrame(labelRootChild, text = "Data Creator")
	label1.grid(row=2, column=1, columnspan=7, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)
	name = StringVar()
	name = Entry(label1, width=28)
	name.grid(row=2, column=1, sticky=W)
	age = IntVar()
	age = Entry(label1, width=28)
	age.grid(row=3, column=1, sticky=W)
	PositionID = IntVar()
	PositionID = Entry(label1, width=28)
	PositionID.grid(row=4, column=1, sticky=W)
	gender = StringVar()
	gender.set("Male")
	rad1 = Radiobutton(label1,text='Male', value="Male", variable=gender)
	rad1.grid(row=5, column=1, sticky=W)
	rad2 = Radiobutton(label1,text='Female', value="Female", variable=gender)
	rad2.grid(row=6, column=1, sticky=W)
	# Text Box Label
	name_label = Label(label1, text="Full Name")
	name_label.grid(row=2, column=0, sticky=W,padx=(10, 10))
	age_label = Label(label1, text="Age")
	age_label.grid(row=3, column=0, sticky=W,padx=(10, 10))
	PositionID_label = Label(label1, text="Position ID")
	PositionID_label.grid(row=4, column=0, sticky=W,padx=(10, 10))
	gender_label = Label(label1, text="Gender")
	gender_label.grid(row=5, column=0, sticky=W,padx=(10, 10))

	# ======================================================== CEK ID FOR ADD DATA =================================================================

	def cek():
		try:
			umur = age.get()
			umur = int(umur)
			position_ = PositionID.get()
			position_ = int(position_)
			# print(name.get())
			src.liveCam.create(name.get(), PositionID.get(), gender.get(), age.get())
			def clearScreen():
				cek_label.grid_forget()
		except ValueError:
			if name.get() == '' or age.get()=='' or PositionID.get()=='':
				messagebox.showinfo("Info", "Data harap diisi lengkap!")
			elif not type(umur) == int:
				messagebox.showinfo("Info", "Age Harus Angka!")
			elif not type(position_) == int:
				messagebox.showinfo("Info", "Position ID Harus Angka!")

	submit = Button(label1, text="submit", command = cek)
	submit.grid(row=7,column=1, ipadx=10, padx=10, sticky=E)

# ================================================== DATABASE ==============================================================================

# INIT
existrecord4 = 0
existrecord3 = 0
existrecord2 = 0
editformexist = 0
existrecord1 = 0
databasePeopleExist = 0
editExist = 0
deleteExist = 0
profileExist = 0
imageexist = 0
databaseAllExist = 0
databasePositionsssExist = 0
editPositionsssLabelFrame = 0
addPositionssLabelFrameExist = 0
editPositionsssFormExist = 0
loginExist = 0

def databaseAll(name):
	global databases
	global databaseAllExist
	reset()

	# ======================================================== DATABASE PEOPLE =================================================================
	def database():
		databaseCopy()
	def databaseCopy():
		resetInsideDb()
		global databasePeopleExist, databasePeopleShow
		databasePeopleShow = LabelFrame(databases, text = "People's")
		databasePeopleShow.grid(row=3, column=1,columnspan=4, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)
		conn = sqlite3.connect("Facebase.db")
		c = conn.cursor()

		databasePeopleExist = 1

		query = "SELECT * From Peoples JOIN Positions USING('PositionID')"
		c.execute(query)
		records = c.fetchall()
		idrecord=''
		namerecord=''
		agerecord=''
		genderrecord=''
		Positionrecord=''
		statusrecord=''
		for record in records:
			idrecord += str(record[0])+"\n"
			namerecord += str(record[2])+"\n"
			agerecord += str(record[3])+"\n"
			genderrecord += str(record[4])+"\n"
			Positionrecord += str(record[5])+"\n"
			statusrecord += str(record[7])+"\n"

		databasePeopleShow2 = LabelFrame(databasePeopleShow, text="Database")
		databasePeopleShow2.grid(row=1, column=0, columnspan = 6, sticky=NW, padx=5, pady=5, ipadx=5, ipady=5)

		main_id = Label(databasePeopleShow2, text="ID")
		main_id.grid(row=1, column=0)
		main_name = Label(databasePeopleShow2, text="Name")
		main_name.grid(row=1, column=1)
		main_age = Label(databasePeopleShow2, text="Age")
		main_age.grid(row=1, column=2)
		main_gender = Label(databasePeopleShow2, text="Gender")
		main_gender.grid(row=1, column=3)
		main_Position = Label(databasePeopleShow2, text="Position")
		main_Position.grid(row=1, column=4)
		main_status = Label(databasePeopleShow2, text="Status")
		main_status.grid(row=1, column=5)
		id_label = Label(databasePeopleShow2, text=idrecord)
		id_label.grid(row=2, column=0)
		name_label = Label(databasePeopleShow2, text=namerecord)
		name_label.grid(row=2, column=1)
		age_label = Label(databasePeopleShow2, text=agerecord)
		age_label.grid(row=2, column=2)
		gender_label = Label(databasePeopleShow2, text=genderrecord)
		gender_label.grid(row=2, column=3)
		Position_label = Label(databasePeopleShow2, text=Positionrecord)
		Position_label.grid(row=2, column=4)
		status_label = Label(databasePeopleShow2, text=statusrecord)
		status_label.grid(row=2, column=5)

		conn.commit()
		conn.close()

		# ======================================================== EDIT DATABASE PEOPLE =================================================================

		def editDbs():
			global editExist, editForm
			resetDb()
			def cekId():
				global existrecord2, query_label2, editFormLabel, editformexist
				try:
					editId = id2.get()
					editId = int(editId)
					if editformexist ==1:
						editFormLabel.grid_forget()
					if existrecord2 == 1:
						query_label2.grid_forget()
					conn = sqlite3.connect('Facebase.db')
					c = conn.cursor()
					query2 = "SELECT * From Peoples JOIN Positions USING('PositionID') WHERE PeopleID="+id2.get()
					c.execute(query2)
					records2 = c.fetchall()
					print_records2=''
					for record2 in records2:
						print_records2 += "ID "+id2.get()+" Detected"
						existrecord2 = 1
					if print_records2 == "":
						query_label2 = Label(editForm, text="ID "+id2.get()+" Empty")
						query_label2.grid(row=1, column=0, columnspan=2, sticky=W)
					else:
						editFormLabel = LabelFrame(editForm, text = "Edit Data")
						editFormLabel.grid(row=5, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
						name = Entry(editFormLabel, width=28)
						name.grid(row=2, column=1, sticky=W)
						name.insert(0,record2[2])
						age = Entry(editFormLabel, width=28)
						age.grid(row=3, column=1, sticky=W)
						age.insert(0,record2[3])
						PositionID = Entry(editFormLabel, width=28)
						PositionID.grid(row=4, column=1, sticky=W)
						PositionID.insert(0,record2[1])
						gender1 = StringVar()
						gender1.set(record2[4])
						rad1 = Radiobutton(editFormLabel,text='Male', value="Male", variable=gender1)
						rad1.grid(row=5, column=1, sticky=W)
						rad2 = Radiobutton(editFormLabel,text='Female', value="Female", variable=gender1)
						rad2.grid(row=6, column=1, sticky=W)
						name_label = Label(editFormLabel, text="Full Name")
						name_label.grid(row=2, column=0, sticky=W,padx=(10, 10))
						age_label = Label(editFormLabel, text="Age")
						age_label.grid(row=3, column=0, sticky=W,padx=(10, 10))
						PositionID_label = Label(editFormLabel, text="Position ID")
						PositionID_label.grid(row=4, column=0, sticky=W,padx=(10, 10))
						gender_label = Label(editFormLabel, text="Gender")
						gender_label.grid(row=5, column=0, sticky=W,padx=(10, 10))
						def Update():
							try:
								if name.get() == '' or age.get()=='' or PositionID.get()=='' or gender1.get()=='':
									messagebox.showinfo("Info", "Data harap diisi lengkap!")
								else:
									umur = age.get()
									umur = int(umur)
									kriminal = PositionID.get()
									kriminal = int(kriminal)
									conn = sqlite3.connect('Facebase.db')
									c = conn.cursor()
									query3 = "UPDATE Peoples SET Name ='"+name.get()+"', PositionID="+PositionID.get()+", Gender='"+gender1.get()+"', Age="+age.get()+" WHERE PeopleID="+id2.get()
									c.execute(query3)
									messagebox.showinfo("Info","ID Updated")
									conn.commit()
									conn.close()
									databasePeopleShow.grid_forget()
									databaseCopy()
							except ValueError:
								if not type(umur) == int:
									messagebox.showinfo("Info", "Age Harus Angka!")
								elif not type(kriminal) == int:
									messagebox.showinfo("Info", "Position ID Harus Angka!")
						editBtn = Button(editFormLabel, text="Update", command = Update)
						editBtn.grid(row=7, column=1, sticky=E)
						query_label2 = Label(editForm, text=print_records2)
						query_label2.grid(row=1, column=0, columnspan=2, sticky=W)
						editformexist = 1
					conn.commit()
					conn.close()
				except ValueError:
					if not type(editId) == int:
						messagebox.showinfo("Info", "ID Harus Angka!")
			editForm = LabelFrame(databasePeopleShow, text = "Edit Form")
			editForm.grid(row=0,column=6, columnspan=6, sticky=NW, padx=5, pady=5, ipadx=5, ipady=5, rowspan=20)
			# Entry ID for Edit
			editExist = 1
			id2 = Entry(editForm)
			id2.grid(row=3, column=1, sticky=W)
			id_label = Label(editForm, text="Masukkan ID")
			id_label.grid(row=3, column=0, sticky=W)
			# Btn Submit
			submitEdit = Button(editForm, text="Submit", command = cekId)
			submitEdit.grid(row=3, column=4, sticky=E, padx=5, pady=5, ipadx=5, ipady=5)
			
		# ======================================================== DELETE DATABASE PEOPLE =================================================================

		def deleteDbs():
			global deleteExist, deleteForm
			resetDb()
			def cekIdDelete():
				global existrecord1, print_records3, query_label2, btnDelete, btnCancel
				try:
					idDelete = id3.get()
					idDelete = int(idDelete)
					
					conn = sqlite3.connect('Facebase.db')			
					c = conn.cursor()
					query2 = "SELECT * From Peoples JOIN Positions USING('PositionID') WHERE PeopleID="+id3.get()
					c.execute(query2)
					record3 = c.fetchall()
					print_records3=''
					for record3 in record3:
						print_records3 += "A"
						query4 = ""
						existrecord1 = 1
					if print_records3 == "":
						messagebox.showinfo("Info", "ID tidak terdaftar dalam database!")
					else:
						MsgBox1 = messagebox.askquestion('Delete','ID '+id3.get()+' will be deleted', icon='warning')
						if MsgBox1 == 'yes':
							conn = sqlite3.connect('Facebase.db')
							c = conn.cursor()
							query5 = "DELETE FROM Peoples WHERE PeopleID="+id3.get()
							c.execute(query5)
							for i in range(1,253):
								if os.path.isdir("./images/3pre_img/"+str(id3.get())) == True:
									shutil.rmtree("./images/3pre_img/"+str(id3.get()))
								if os.path.isdir("./images/1train_img/"+str(id3.get())) == True:
									shutil.rmtree("./images/1train_img/"+str(id3.get()))
								if os.path.isdir("./images/2align_face/"+str(id3.get())) == True:
									shutil.rmtree("./images/2align_face/"+str(id3.get()))
							deleteForm.grid_forget()
							conn.commit()
							conn.close()
							messagebox.showinfo("Info","Delete successful!")
							deleteForm.grid_forget()
							databaseCopy()
							MsgBox2 = messagebox.askquestion('Delete','Train to avoid future error!', icon='warning')
							if MsgBox2 == 'yes':
								train_image()
						def cancelDelete():
							deleteForm.grid_forget()
				except ValueError:
					if id3.get() == '':
						messagebox.showinfo("Info", "ID belum terisi!")
					if not type(idDelete) == int and not id3.get() == '':
						messagebox.showinfo("Info", "ID harus angka!")
			deleteExist = 1
			deleteForm  = LabelFrame(databasePeopleShow, text="Delete Form")
			deleteForm.grid(row=0, column=6, columnspan=6, sticky=NW, padx=5, pady=5, ipadx=5, ipady=5, rowspan=20)
			id3 = Entry(deleteForm)
			id3.grid(row=0, column=1, sticky=W)
			id3_label = Label(deleteForm, text="Masukkan ID")
			id3_label.grid(row=0, column=0, sticky=W)

			cekDelete = Button(deleteForm, text = "Delete", command = cekIdDelete)
			cekDelete.grid(row=0, column=2, sticky=E, padx=5, pady=5, ipadx=5, ipady=5)

		# ======================================================== DATABASE PEOPLE IMAGES =================================================================

		def imageDbs():
			global imageForm, imageexist
			
			resetDb()

			imageexist = 1
			def cekImages():
				global profileExist, existrecord3, query_label3, profile
				try:
					idImage = id4.get()
					idImage = int(idImage)
					if profileExist == 1:
						profile.grid_forget()

					conn = sqlite3.connect('Facebase.db')			
					c = conn.cursor()
					query = "SELECT * From Peoples JOIN Positions USING('PositionID') WHERE PeopleID="+id4.get()
					c.execute(query)
					records = c.fetchall()
					print_records4=''

					for record in records:
						print_records4= "ada"
					if print_records4 == "":
						messagebox.showinfo("Info", "ID "+id4.get()+" belum ada dalam database")

					else:
						profile = LabelFrame(imageForm, text="Profile")
						profile.grid(row=1,columnspan=3)
						load = Image.open("./images/2align_face/"+id4.get()+"/1.png")
						load = load.resize((250,250), Image.ANTIALIAS)
						render = ImageTk.PhotoImage(load)
						img = Label(profile, image=render)
						img.image = render
						img.grid(row=0, column=0,columnspan=4)
						profileExist=1
						name = Label(profile, text="Name ")
						name.grid(row=1, column=0, sticky=W)
						Age = Label(profile, text="Age ")
						Age.grid(row=2, column=0, sticky=W)
						Gender = Label(profile, text="Gender ")
						Gender.grid(row=3, column=0, sticky=W)
						Position = Label(profile, text="Position ")
						Position.grid(row=4, column=0, sticky=W)
						Status = Label(profile, text="Status ")
						Status.grid(row=5, column=0, sticky=W)
						name2 = Label(profile, text=":   "+str(record[2]))
						name2.grid(row=1, column=1, sticky=W)
						Age2 = Label(profile, text=":   "+str(record[3]))
						Age2.grid(row=2, column=1, sticky=W)
						Gender2 = Label(profile, text=":   "+str(record[4]))
						Gender2.grid(row=3, column=1, sticky=W)
						Position2 = Label(profile, text=":   "+str(record[5]))
						Position2.grid(row=4, column=1, sticky=W)
						Status2 = Label(profile, text=":   "+str(record[7]))
						Status2.grid(row=5, column=1, sticky=W)

					existrecord3 = 1
				except ValueError:
					if not type(idImage) == int:
						messagebox.showinfo("Info", "ID Harus Angka!")

			imageForm = LabelFrame(databasePeopleShow, text="Search Form")
			imageForm.grid(row=0, column=6, columnspan=6, sticky=NW, rowspan=20, padx=5, pady=5, ipady=5)
			id4 = Entry(imageForm)
			id4.grid(row=0, column=1, sticky=W)
			id4_label = Label(imageForm, text="Masukkan ID")
			id4_label.grid(row=0, column=0, sticky=W)

			cekImage = Button(imageForm, text = "Submit", command = cekImages)
			cekImage.grid(row=0, column=2, sticky=E, padx=5, pady=5, ipadx=5, ipady=5)

		# ======================================================== GUI DATABASE PEOPLE =================================================================

		editDb = Button(databasePeopleShow, text="Edit", command = editDbs)
		editDb.config(width=10)
		editDb.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky=NW)
		deleteDb = Button(databasePeopleShow, text="Delete", command = deleteDbs)
		deleteDb.config(width=10)
		deleteDb.grid(row=0, column=2, columnspan=2, padx=10, pady=5, sticky=N)
		imageDb = Button(databasePeopleShow, text="Search", command = imageDbs)
		imageDb.config(width=10)
		imageDb.grid(row=0, column=4,columnspan=2, padx=10, pady=5, sticky=NE)

	# ======================================================== DATABASE Positionsss =================================================================

	def databasePositionsss():
		databasePositionsssCopy()

	def databasePositionsssCopy():
		global databasePositionsssExist, databasePositionsshow

		resetInsideDb()
		databasePositionsshow = LabelFrame(databases, text= "Position's")
		databasePositionsshow.grid(row=2, column=1,columnspan=4, rowspan=4,sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)
		conn = sqlite3.connect("Facebase.db")
		c = conn.cursor()
		query = "SELECT * From Positions"
		c.execute(query)
		records = c.fetchall()
		PositionIDs=''
		types=''
		levels=''
		statuss=''
		warnas=''
		for record in records:
			PositionIDs += str(record[0])+"\n"
			types += str(record[1])+"\n"
			levels += str(record[2])+"\n"
			statuss += str(record[3])+"\n"
			warnas += str(record[4])+"\n"

		databasePositionsshow2 = LabelFrame(databasePositionsshow, text= "Database")
		databasePositionsshow2.grid(row=1, column=0, columnspan = 6, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)

		PositionID = Label(databasePositionsshow2, text="Position ID")
		PositionID.grid(row=0, column=0)
		thetype = Label(databasePositionsshow2, text="Type")
		thetype.grid(row=0, column=1)
		level = Label(databasePositionsshow2, text="Level")
		level.grid(row=0, column=2)
		status = Label(databasePositionsshow2, text="Status")
		status.grid(row=0, column=3)
		warna = Label(databasePositionsshow2, text="Color")
		warna.grid(row=0, column=4)
		
		PositionID_label = Label(databasePositionsshow2, text=PositionIDs)
		PositionID_label.grid(row=1, column=0)
		thetype_label = Label(databasePositionsshow2, text=types)
		thetype_label.grid(row=1, column=1)
		level_label = Label(databasePositionsshow2, text=levels)
		level_label.grid(row=1, column=2)
		status_label = Label(databasePositionsshow2, text=statuss)
		status_label.grid(row=1, column=3)
		warna_label = Label(databasePositionsshow2, text=warnas)
		warna_label.grid(row=1, column=4)

		conn.commit()
		conn.close()

		def addPositionss():
			global addPositionssLabelFrame, addPositionssLabelFrameExist
			resetDc()
			def cekAddPositionss():
				def submitPositionsss():
					conn = sqlite3.connect('Facebase.db')
					c = conn.cursor()
					query = "INSERT INTO Positions(Type, Level, Status, Warna) VALUES ('"+addType.get()+"', "+addLevel.get()+", '"+addStatus.get()+"', '"+addColor.get()+"' )"
					c.execute(query)
					conn.commit()
					conn.close()
					
				if addType.get()=='' or addLevel.get()=='' or addStatus.get()=='' or addColor.get()=='':
					messagebox.showinfo("Info", "Data harap diisi lengkap!")
				else:
					submitPositionsss()
			Statusss = [
			"Student",
			"Teacher",
			"TU",
			"Orang Asing",
			"OB",]

			addPositionssLabelFrame = LabelFrame(databasePositionsshow, text = "Add Positionss")
			addPositionssLabelFrame.grid(row=0, column=6, columnspan=7, rowspan=10,sticky=NW, padx=5, pady=5, ipadx=5, ipady=5)
			addType = Entry(addPositionssLabelFrame, width=28)
			addType.grid(row=2, column=1, sticky=W, columnspan=2)
			addLevel = Entry(addPositionssLabelFrame, width=28)
			addLevel.grid(row=3, column=1, sticky=W, columnspan=2)
			addStatusVar = StringVar()
			addStatusVar.set(Statusss[0])
			addStatus = OptionMenu(addPositionssLabelFrame, addStatusVar, *Statusss)
			addStatus.config(width=22)
			addStatus.grid(row=4, column=1, columnspan=2, sticky=W)
			addColor = StringVar()
			addColor.set("Green")

			rad1 = Radiobutton(addPositionssLabelFrame,text='Green', value="Green", variable=addColor)
			rad1.grid(row=5, column=1, sticky=W)
			rad2 = Radiobutton(addPositionssLabelFrame,text='Red', value="Red", variable=addColor)
			rad2.grid(row=6, column=1, sticky=W)
			rad3 = Radiobutton(addPositionssLabelFrame,text='Blue', value="Blue", variable=addColor)
			rad3.grid(row=5, column=2, sticky=W)
			rad4 = Radiobutton(addPositionssLabelFrame,text='Black', value="Black", variable=addColor)
			rad4.grid(row=6, column=2, sticky=W)

			# Text Box Label
			addTypelabel = Label(addPositionssLabelFrame, text="Type")
			addTypelabel.grid(row=2, column=0, sticky=W,padx=(10, 10))
			addLevellabel = Label(addPositionssLabelFrame, text="Level")
			addLevellabel.grid(row=3, column=0, sticky=W,padx=(10, 10))
			addStatuslabel = Label(addPositionssLabelFrame, text="Status")
			addStatuslabel.grid(row=4, column=0, sticky=W,padx=(10, 10))
			addColorlabel = Label(addPositionssLabelFrame, text="Warna")
			addColorlabel.grid(row=5, column=0, sticky=W,padx=(10, 10))

			submit = Button(addPositionssLabelFrame, text="submit", command = cekAddPositionss)
			submit.grid(row=7,column=2, ipadx=10, pady=5, sticky=E)
			addPositionssLabelFrameExist = 1

		def editPositionss():
			global editPositionsssFormExist, editPositionsssForm
			resetDc()
			def cekId():
				global editPositionsssLabelFrame, addPositionssLabelFrame, existrecord4, query_label3

				if editPositionsssLabelFrame == 1:
					addPositionssLabelFrame.grid_forget()
				if existrecord4 == 1:
					query_label3.grid_forget()

				conn = sqlite3.connect('Facebase.db')
				c = conn.cursor()
				query2 = "SELECT * From Positions WHERE PositionID="+id2.get()
				c.execute(query2)
				records2 = c.fetchall()
				print_records2=''

				for record2 in records2:
					print_records2 += "ID "+id2.get()+" Detected"
					existrecord2 = 1

				if print_records2 == "":
					query_label3 = Label(addPositionssLabelFrame, text="ID "+id2.get()+" Empty")
					query_label3.grid(row=1, column=0, columnspan=2, sticky=W)
				else:
					addPositionssLabelFrame = LabelFrame(editPositionsssForm, text = "Edit Positionss")
					addPositionssLabelFrame.grid(row=4, column=0, columnspan=7, rowspan=10,sticky=NW, padx=5, pady=5, ipadx=5, ipady=5)
					addType = Entry(addPositionssLabelFrame, width=28)
					addType.grid(row=2, column=1, sticky=W, columnspan=2)
					addType.insert(0,record2[1])
					addLevel = Entry(addPositionssLabelFrame, width=28)
					addLevel.grid(row=3, column=1, sticky=W, columnspan=2)
					addLevel.insert(0,record2[2])
					addStatus = Entry(addPositionssLabelFrame, width=28)
					addStatus.grid(row=4, column=1, sticky=W, columnspan=2)
					addStatus.insert(0,record2[3])
					addColor = StringVar()
					addColor.set(record2[4])

					rad1 = Radiobutton(addPositionssLabelFrame,text='Green', value="Green", variable=addColor)
					rad1.grid(row=5, column=1, sticky=W)
					rad2 = Radiobutton(addPositionssLabelFrame,text='Red', value="Red", variable=addColor)
					rad2.grid(row=6, column=1, sticky=W)
					rad3 = Radiobutton(addPositionssLabelFrame,text='Blue', value="Blue", variable=addColor)
					rad3.grid(row=5, column=2, sticky=W)
					rad4 = Radiobutton(addPositionssLabelFrame,text='Black', value="Black", variable=addColor)
					rad4.grid(row=6, column=2, sticky=W)

					addPositionIDLabel = Label(addPositionssLabelFrame, text="Position ID")
					addPositionIDLabel.grid(row=1, column=0, sticky=W,padx=(10, 10))
					addTypelabel = Label(addPositionssLabelFrame, text="Type")
					addTypelabel.grid(row=2, column=0, sticky=W,padx=(10, 10))
					addLevellabel = Label(addPositionssLabelFrame, text="Level")
					addLevellabel.grid(row=3, column=0, sticky=W,padx=(10, 10))
					addStatuslabel = Label(addPositionssLabelFrame, text="Status")
					addStatuslabel.grid(row=4, column=0, sticky=W,padx=(10, 10))
					addColorlabel = Label(addPositionssLabelFrame, text="Warna")
					addColorlabel.grid(row=5, column=0, sticky=W,padx=(10, 10))

					def Update():
						conn = sqlite3.connect('Facebase.db')
						c = conn.cursor()
						query3 = "UPDATE Positions SET Type ='"+addType.get()+"', Level="+addLevel.get()+", Status='"+addStatus.get()+"', Warna='"+addColor.get()+"' WHERE PositionID="+id2.get()
						c.execute(query3)
						messagebox.showinfo("Info","ID Updated")
						conn.commit()
						conn.close()
						databasePositionsshow.grid_forget()
						databasePositionsssCopy()
					submit = Button(addPositionssLabelFrame, text="submit", command = Update)
					submit.grid(row=7,column=2, ipadx=10, pady=5, sticky=E)

					query_label2 = Label(addPositionssLabelFrame, text=print_records2)
					query_label2.grid(row=1, column=0, columnspan=2, sticky=W)
					editPositionsssLabelFrame = 1

				conn.commit()
				conn.close()

			editPositionsssForm = LabelFrame(databasePositionsshow, text = "Edit Form")
			editPositionsssForm.grid(row=0,column=6, columnspan=6, sticky=NW, padx=5, pady=5, ipadx=5, ipady=5, rowspan=6)
			# Entry ID for Edit
			editExist = 1
			id2 = Entry(editPositionsssForm)
			id2.grid(row=3, column=1, sticky=W)
			id_label = Label(editPositionsssForm, text="Masukkan ID")
			id_label.grid(row=3, column=0, sticky=W)
			# Btn Submit
			submitEdit = Button(editPositionsssForm, text="Submit", command = cekId)
			submitEdit.grid(row=3, column=4, sticky=E, padx=5, pady=5, ipadx=5, ipady=5)
			editPositionsssFormExist = 1

		def deletePositionss():
			return

		addPositionsss = Button(databasePositionsshow, text="Add", command = addPositionss)
		addPositionsss.grid(row=0, column=0, columnspan=4, ipadx=10, padx=5, pady=5, sticky=NW)
		editPositionsss = Button(databasePositionsshow, text="Edit", command = editPositionss)
		editPositionsss.grid(row=0, column=4, columnspan=2, ipadx=10,  padx=5, pady=5, sticky=NE)
		databasePositionsssExist = 1

	# ======================================================== GUI DATABBASE ==================================================

	databases = LabelFrame(labelRootChild, text = "DBMS")
	databases.grid(row=1, column=1,columnspan=4, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)

	databasePeople = Button(databases, text="People", command = database)
	databasePeople.config(width=10)
	databasePeople.grid(row=1, column=1, sticky=NW, padx=2)
	databasePositionsss = Button(databases, text="Position", command = databasePositionsss)
	databasePositionsss.config(width=10)
	databasePositionsss.grid(row=1, column=4, sticky=NE, padx=2)
	databaseAllExist = 1
	messagebox.showinfo("Info", "Welcome "+str(name))


def loginss():
	global loginExist, logins
	reset()
	loginExist = 1
	logins = LabelFrame(labelRootChild, text = "Login")
	logins.grid(row=1, column=1,columnspan=4, sticky='NW', padx=5, pady=5, ipadx=5, ipady=5)
	def auth(event):
		with sqlite3.connect("Facebase.db") as db:
			cursor = db.cursor()
		find_user = ("SELECT * FROM Users WHERE username = ? AND password = ?")
		cursor.execute(find_user,[(usernames.get()),(passwords.get())])
		result = cursor.fetchall()
		if result:
			databaseAll(result[0][0])
		else:
			messagebox.showinfo("Info", "Username & Password incorrect!")

	username = Label(logins, text="Username")
	username.grid(row=1, column=0, sticky=W)
	usernames = Entry(logins, width=28)
	usernames.grid(row=1, column=1, sticky=W)
	password = Label(logins, text="Password")
	password.grid(row=2, column=0, sticky=W)
	passwords = Entry(logins, show="*", width=28)
	passwords.grid(row=2, column=1, sticky=W)
	passwords.bind('<Return>', auth)
# ================================================= EXIT WINDOWS ===========================================================================

def exitWindow():
	MsgBox1 = messagebox.askquestion('Trainer',' Are you sure you want to run quit the program?', icon='warning')
	if MsgBox1 == 'yes':
		root.destroy()

# ===================================================== GUI MENU ===========================================================================

mainFrame = Frame(root)
mainFrame.grid(sticky=NW)

mainFrame.grid_rowconfigure(0, weight=1)
mainFrame.grid_columnconfigure(0, weight=1)

labelRoot = LabelFrame(mainFrame, text="MENU")
labelRoot.grid(row=0, column=0, rowspan=50, sticky='NW', padx=5, pady=5, ipady=5)

detector = Button(labelRoot, text="DETECTOR", command = detector, anchor=CENTER)
detector.config(width = 25)
detector.grid(row=0, column=0, pady=10, padx=10, ipady=10)

trainer = Button(labelRoot, text=" TRAINER ", command = trainer, anchor=CENTER)
trainer.grid(row=1, column=0, pady=10, padx=10, ipady=10)
trainer.config(width = 25)

creator = Button(labelRoot, text="CREATOR", command = creator, anchor=CENTER)
creator.grid(row=2, column=0, pady=10, padx=10, ipady=10)
creator.config(width = 25)

database = Button(labelRoot, text="DATABASE", command = loginss, anchor=CENTER)
database.grid(row=3, column=0, pady=10, padx=10, ipady=10)
database.config(width = 25)

exit = Button(labelRoot, text="EXIT", command = exitWindow, anchor=CENTER)
exit.grid(row=5, column=0, pady=10, padx=10, ipady=10)
exit.config(width = 25)

labelRootChild = LabelFrame(mainFrame, text="SUB MENU")
labelRootChild.grid(row=0, column=1, sticky='NE', padx=5, pady=5, ipady=5)

# ======================================================		GUI MAINLOOP  	===============================================================

root.mainloop()

# ================================================			END OF THE CODES		================================================================