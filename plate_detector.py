# Detect vehicles
#python vehicle-detection.py $input_dir $output_dir

# Detect license plates
#python license-plate-detection.py $output_dir $lp_model

# OCR
#python license-plate-ocr.py $output_dir

# Draw output and generate list
#python gen-outputs.py $input_dir $output_dir > $csv_file

import os
import sys
import cv2
import numpy as np
import traceback


from os.path import splitext, basename, isdir, isfile
from os import makedirs
from pdb import set_trace as pause
from glob import glob

import alpr_unconstrained.darknet.python.darknet as dn
from alpr_unconstrained.darknet.python.darknet import detect
from alpr_unconstrained.src.label import dknet_label_conversion
from alpr_unconstrained.src.utils import nms, crop_region, image_files_from_folder
from alpr_unconstrained.src.utils import im2single
from alpr_unconstrained.src.label import Label, lwrite, Shape, writeShapes, lread, readShapes
from alpr_unconstrained.src.keras_utils import load_model, detect_lp
from alpr_unconstrained.src.drawing_utils import draw_label, draw_losangle, write2img
from plate_analysis_tool import plate_analysis


from configurations.config import ConfigParser
import sys,os
config_obj = ConfigParser()

#sys.path.append(os.getcwd())

#########################################################################
# car recogonize

vehicle_threshold = .5
#vehicle_weights = bytes(config_obj.vehicle_weights, encoding="utf-8")
#vehicle_netcfg  = bytes(config_obj.vehicle_netcfg, encoding="utf-8")
#vehicle_dataset = bytes(config_obj.vehicle_dataset, encoding="utf-8")
vehicle_weights = bytes(config_obj.vehicle_weights)
vehicle_netcfg  = bytes(config_obj.vehicle_netcfg)
vehicle_dataset = bytes(config_obj.vehicle_dataset)
vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)
#########################################################################
# plate recogonize

lp_threshold = .5
wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
#wpod_net_path = sys.argv[2]
wpod_net = load_model(wpod_net_path)

#########################################################################
# plate charactor recogonize
ocr_weights = bytes(config_obj.ocr_weights)
ocr_netcfg  = bytes(config_obj.ocr_netcfg)
ocr_dataset = bytes(config_obj.ocr_dataset)


ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)
ocr_threshold = .1


class Plate_Detector():

    def __init__(self):
		pass

    def vehicle_detector(self, img_path):
		"""
		Detect a vehicle, rewrite of vehicle-detection.py
		"""
		#img_path = "alpr_unconstrained/samples/test/03009.jpg"
		#img_path = "test_car_vbot_4.png"
		#img_path =  bytes(img_path, encoding="utf-8")
		#net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
		#meta = load_meta("cfg/coco.data")
		img_path =  bytes(img_path)
		R,_ = detect(vehicle_net, vehicle_meta, img_path, thresh=vehicle_threshold)

		R = [r for r in R if r[0].decode(encoding='UTF-8') in ['car','bus']]

		print('\t\t%d cars found' % len(R))
		if len(R):
			img_path = img_path.decode(encoding='UTF-8')
			Iorig = cv2.imread(img_path)
			WH = np.array(Iorig.shape[1::-1],dtype=float)
			Lcars = []

			for i,r in enumerate(R):

				#cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
				cx,cy,w,h = (np.array(r[2])).tolist()
				x1 = int(cx - w/2.)
				y1 = int(cy - h/2.)
				x2 = int(cx + w/2.)
				y2 = int(cy + h/2.)
				tl = np.array([x1, y1])
				br = np.array([x2, y2])
				label = Label(0,tl,br)
				
				#Icar = crop_region(Iorig,label)
				cv2.rectangle(Iorig, (x1, y1), (x2, y2), (255,0,0), 2)
				#cv2.putText(Iorig,'Auto Detected',(x2+10,y2+h),0,0.3,(0,255,0))

				Lcars.append(label)
				print(label)
			return Lcars, Iorig, WH
		else:
			return [], None, None
		
    def plate_recognizer(self, img_path):
		#imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
		print('Performing OCR...')
		#img_path = 'data_output//test_lp.png'
		#img_path = bytes(img_path, encoding="utf-8")
		img_path = bytes(img_path)
		R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, hier_thresh=ocr_threshold, nms=None, path_mode = True)
		#R,(width,height) = detect(ocr_net, ocr_meta, im.fromarray(Ilp*255., 'RGB')   ,thresh=ocr_threshold, nms=None, path_mode = False)

		if len(R):
			L = dknet_label_conversion(R,width,height)
			L = nms(L,.1)

			L.sort(key=lambda x: x.tl()[0])
			lp_str = ''.join([chr(l.cl()) for l in L])

			#with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
			#f.write(lp_str + '\n')

			print('\t\tLP: %s' % lp_str)
			return lp_str
		else:
			print('No characters found')
			return None

    def plate_detector(self, Ivehicle, output_dir, bname):
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
		plat_image_path = []
		if len(LlpImgs):
			i = 0
			for Ilp in LlpImgs:
				#Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
				#s = Shape(Llp[0].pts)
				image_path = '%s/%s_lp.png' % (output_dir,bname + str(i-1))
				cv2.imwrite(image_path, Ilp*255.)
				#writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
				i +=1
				plat_image_path.append(image_path)
			return Llp, plat_image_path
		else:
			return None, plat_image_path

    def car_filter(self, Lcars, Llp_i, WH):
		if len(Lcars)==0:
			return False
		lp_tl_x,lp_tl_y = Llp_i.tl()
		lp_br_x,lp_br_y = Llp_i.br()
		w = WH[0]
		h = WH[1]
		lp_tl_x = lp_tl_x*w
		lp_tl_y = lp_tl_y*h
		lp_br_x = lp_br_x*w
		lp_br_y = lp_br_y*h
		
		for lcari in Lcars:
			tl_x, tl_y = lcari.tl()
			br_x, br_y = lcari.br()
			#print( (tl_x > lp_tl_x) , (tl_y > lp_tl_y))
			if (tl_x < lp_tl_x) and (tl_y < lp_tl_y):
				if (br_x > lp_br_x) and (br_y > lp_br_y):
					return True
		return False
				

    def video_scaner(self, video_path, output_path,
					 time_break = 10, number_frame = 1500):
		self.time_break = time_break
		i = 0
		#'video/ALRP_left_zone_A_wo_ir_dots.avi'
		cap = cv2.VideoCapture(video_path)

		plat_snap_shots = []
		if not os.path.exists(output_path):
			os.makedirs(output_path)
			os.makedirs(output_path+"/screen_shot")
			os.makedirs(output_path+"/plate_image")

		while(cap.isOpened()):	
			ret, frame = cap.read()
			print("For Frame: ", i-1)
			if i >= number_frame-1:
				break
			if i%time_break!=0:
				i = i+1
			else:
				i = i+1

				print("For Frame: ", i-1)
				#im = cv2.resize(frame,(640,480))
				img_path = output_path + '/screen_shot/test_car_vbot_' + str(i-1) + '.png'
				check = cv2.imwrite(img_path ,frame)
				if check == False:
					continue
				#print("start detect vehicle")
				Lcars, Iorig, WH = self.vehicle_detector(img_path)

				if len(Lcars)==0:
					os.remove(img_path)
					continue
				else:
					# missing car location filter
					plat_path = output_path + '/plate_image'
					bname = str(i-1) + "_plate_number_"
					#print("start detect plate")
					Llp, plat_image_path = self.plate_detector(Iorig, plat_path, bname)
					if Llp == None:
						os.remove(img_path)
						continue
					plat_str_dict = {}
					plat_str_dict["time"] = i-1
					plat_str_dict["plat"] = []
					plat_str_dict["Lcars"] = Lcars
					#print("start read plate")
					lp_str = ""
					for Llpi, plat_path_i in zip(Llp, plat_image_path):
						if self.car_filter(Lcars, Llp[0], WH):
							lp_str = self.plate_recognizer(plat_path_i)
							plat_str_dict["plat"].append({"lp_str":lp_str,
												"lp_loc": Llpi})
						else:
							os.remove(plat_path_i)
							continue 

					if lp_str == "":
						os.remove(img_path)

						continue 
						
					plat_snap_shots.append(plat_str_dict)
			
		cap.release()
		cv2.destroyAllWindows()
		print("Finished")
		return plat_snap_shots
	
    def plats_summary(self, plat_snap_shots, diff_i = 0.12):
		time_break = self.time_break 
		dedup_plats = plate_analysis(plat_snap_shots, time_break=time_break, diff_i = diff_i)
		return dedup_plats






				
				

		



		



