from imutils.face_utils import FaceAligner
import os
import numpy as np
import src.facenet
import dlib
import cv2

class align:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = src.facenet.get_dataset(self.input_datadir)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./class/shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor)

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = cv2.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            rects = detector(gray)
                            for face in rects:
                                    top = max(0, face.top())
                                    bottom = min(face.bottom(), img.shape[0])
                                    left = max(0, face.left())
                                    right = min(face.right(), img.shape[1])
                                    faceBoxRectangleS = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
                                    faceAligned = fa.align(img, gray, faceBoxRectangleS)
                                    cv2.imwrite(output_filename, faceAligned)
                                    nrof_successfully_aligned+=1
        return (nrof_images_total,nrof_successfully_aligned)