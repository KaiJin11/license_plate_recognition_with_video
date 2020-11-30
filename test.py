from plate_detector import Plate_Detector
pdetect_obj = Plate_Detector()

video_path = 'video/ALRP_left_zone_A_wo_ir_dots.avi'
output_path = 'data_output'
pdetect_obj.video_scaner(video_path, output_path)
