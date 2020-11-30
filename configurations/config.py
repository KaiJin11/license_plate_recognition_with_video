import os 
import configparser


class ConfigParser():
    """
    Configuration class for serving data
    """
    def __init__(self):
        config = configparser.RawConfigParser()
        file_path = "config.ini"
        config.readfp(open(file_path))

        self.version = config.get("Version", "VERSION")
        self.version = config.get("Version", "VERSION")
        self.version = config.get("Version", "VERSION")

        self.vehicle_weights = config.get("Model_Path", "vehicle_weights")
        self.vehicle_netcfg = config.get("Model_Path", "vehicle_netcfg")
        self.vehicle_dataset = config.get("Model_Path", "vehicle_dataset")
        self.ocr_weights = config.get("Model_Path", "ocr_weights")
        self.ocr_netcfg = config.get("Model_Path", "ocr_netcfg")
        self.ocr_dataset = config.get("Model_Path", "ocr_dataset")
        self.darknet = config.get("Model_Path", "darknet")
