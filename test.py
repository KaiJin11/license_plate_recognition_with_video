from plate_detector import Plate_Detector
pdetect_obj = Plate_Detector()

video_path = 'video/ALRP_left_zone_A_wo_ir_dots.avi' #1500
output_path = 'data_output_2'
result_plates = pdetect_obj.video_scaner(video_path, output_path, time_break = time_break, number_frame = 1500)


dedup_plats = pdetect_obj.plats_summary(result_plates)
dedup_plats
