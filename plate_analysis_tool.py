import sys
import itertools
import numpy as np
from collections import Counter 

def center_extractor(loc1 ):
    x = (loc1.br()[0] + loc1.tl()[0])/2
    y = (loc1.br()[1] + loc1.tl()[1])/2
    return(x,y)

def size_extractor(loc1):
    x = (loc1.br()[0] - loc1.tl()[0])
    y = (loc1.br()[1] - loc1.tl()[1])
    return(x*y)

def center_extractor(loc1 ):
    x = (loc1.br()[0] + loc1.tl()[0])/2
    y = (loc1.br()[1] + loc1.tl()[1])/2
    return(x,y)

def check_existing_plats(exisitng_plats, plat, time, center, time_break):
    center_j = center_extractor(plat['lp_loc']) 
    for plates_id, plat_dict in exisitng_plats.items():
        max_time = max(plat_dict["time"])
        if time - max_time > time_break*2:
            continue
        for center in plat_dict["plate_center"]:
            dist = np.linalg.norm(np.array([center]) - np.array([center_j])) 
            if dist < 0.12:
                return plates_id
    return None

def update_plates(exisitng_plats, adding_plats, plates_id, time, time_break):
    if len(adding_plats) > 0:
        for i in range(len(adding_plats)):
            pi = dict(adding_plats[i])
            center = center_extractor(pi['lp_loc'])
            matching_id = check_existing_plats(exisitng_plats, pi, time, center, time_break)

            if matching_id == None:
                exisitng_plats[plates_id] = {}
                exisitng_plats[plates_id]["plates"] = [pi]
                exisitng_plats[plates_id]["time"] = [time]
                exisitng_plats[plates_id]["plate_center"] = [center]
                plates_id += 1
            else:
                exisitng_plats[matching_id]["plates"].append(pi)
                exisitng_plats[matching_id]["time"].append(time)
                exisitng_plats[matching_id]["plate_center"].append(center)

    return exisitng_plats, plates_id

def plate_analysis(result_plates, time_break, diff_i = 0.12):

    previous_plats = result_plates[0]['plat']
    previous_time =  result_plates[0]['time']

    centers_distense = []
    size_difference = []
    plate_size = []
    time_list = []
    plates_str = []
    plates_shift = []

    time = previous_time
    plates_id = 0
    exisitng_plats = {}
    exisitng_plats, plates_id = update_plates(exisitng_plats, previous_plats, plates_id, time, time_break)
    
    for data in result_plates:
        plats = data['plat']
        time =  data['time']
        if time - previous_time > time_break:
            
            previous_plats = plats
            previous_time = time
            exisitng_plats, plates_id = update_plates(exisitng_plats, previous_plats, plates_id, time, time_break)
            continue
        
        
        for plat in plats:
            if plat['lp_str'] == None:
                continue

            center_i = center_extractor(plat['lp_loc'])
            size_i = size_extractor(plat['lp_loc'])
            
            for plat_j in previous_plats:

                if plat_j['lp_str'] == None:
                    continue
                
                center_j = center_extractor(plat_j['lp_loc'])
                size_j = size_extractor(plat_j['lp_loc'])    
                
                dist = np.linalg.norm(np.array([center_i]) - np.array([center_j]) ) 
                size_diff = np.abs(size_i - size_j)
                time_list.append(time)
                centers_distense.append(dist)
                size_difference.append(size_diff)
                plate_size.append(size_i)
                plates_str.append(str(time) + " : " + plat_j['lp_str'] + "==>" + plat['lp_str'] )
                plates_shift.append((plat_j['lp_str'] , plat['lp_str'] ))        
                exisitng_plats, plates_id = update_plates(exisitng_plats, [plat], plates_id, time, time_break)  

                
                
        previous_plats = plats
        previous_time = time
    
    #time_break = 5 
    unique_plates = {}
    previous_plate = {}
    start_id = 0
    for label, x, y, si in zip(plates_shift, time_list, centers_distense, plate_size):
        if x!=start_id:
            #print(start_id, previous_plate)
            for pl0, pl1 in previous_plate.items():
                additional_dict = {"plate": pl1, "time": start_id, "size": si}
                all_plat_list = [v["Past Plates"] for _, v in unique_plates.items()] 
                all_plat_list = list(itertools.chain.from_iterable(all_plat_list))
                if pl0 not in [pi["plate"] for pi in all_plat_list]:
                    unique_plates[len(unique_plates)] = {"Predicted Plates":"None", 
                                                        "Past Plates":[additional_dict]}
                
                for ind, plat_list in unique_plates.items():
                    
                    #plat_list = list(itertools.chain.from_iterable(plat_list))
                    if pl0 in [pi["plate"] for pi in plat_list["Past Plates"]]:
                        unique_plates[ind]["Past Plates"].append(additional_dict)
                    #else:
                    #    unique_plates[len(unique_plates)] = [additional_dict]
                        
            start_id = x
            previous_plate = {}
            

        if y < diff_i: 
            if previous_plate.get(label[0]) == None:
                previous_plate[label[0]] = label[1]
    
    plate_valid = []
    plate_size = []
    valid_plate_length = 7
    for ind, plats_dict in unique_plates.items():
        pps = plats_dict['Past Plates']
        valid_plate_length_check = False
        if 7 in [len(pp["plate"]) for pp in pps]:
            valid_plate_length_check = True
            
        valid_plates = []
        for pp in pps:
            plate = pp["plate"]
            size = pp["size"]
            if valid_plate_length_check:
                if len(plate) != valid_plate_length:
                    continue
            if len(plate) < 5:
                continue
            valid_plates.append(plate)
        valid_plates = Counter(valid_plates).most_common()
        sum_counts = float(sum([score[1] for score in valid_plates]))
        valid_plates = [(pi, si/sum_counts) for pi,si in valid_plates]
        unique_plates[ind]["Predicted Plates"] = valid_plates
        #print(valid_plates)
    return unique_plates
                        