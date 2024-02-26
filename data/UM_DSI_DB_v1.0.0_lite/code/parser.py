from datetime import datetime
import re

"""
This file has the purpose of parsing the dataset files into Python structures such as Lists or Dicts.

"""


def sorted_nicely(lst):
    """
    Sort given iterable alphanumerically.

    :param lst: List of strings to sort
    :return: Sorted list
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)


def parse_dataset(folder_path):
    """
    Parse the dataset information into a Python dictionary with the following format
    dataset = {
        "timestamps": [ datetime1, datetime2, ..., datetimeN ],
        "coordinates": [ [x1,y1,z1], [x2,y2,z2], ..., [xN,yN,zN]],
        "rssis": [
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint 1
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint 2
            ...
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint N
         ],
        "channels": [
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint 1
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint 2
            ...
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint N
         ]
        }

    :param folder_path: Path of the dataset folder
    :return: Dict with dataset
    """
    dataset = dict()
    dataset['timestamps'] = parse_timestamps(folder_path / "timestamps.csv")
    dataset['coordinates'] = parse_coordinates(folder_path / "coordinates.csv")
    dataset['rssis'] = parse_info_by_ap(folder_path / "rssis.csv")
    dataset['channels'] = parse_info_by_ap(folder_path / "channels.csv")

    print("Parsed " + str(len(dataset['timestamps'])) + " timestamps;")
    print("Parsed " + str(len(dataset['coordinates'])) + " coordinates;")
    print("Parsed " + str(len(dataset['rssis'])) + " rssis;")
    print("Parsed " + str(len(dataset['channels'])) + " channels;")

    return dataset


def parse_timestamps(file_path):
    """
    Parse csv file with list of timestamps.
    All timestamps are converted into datetime objects.

    :param file_path: Path where file is located
    :return: list of datetime timestamps
    """
    timestamps = list()
    with open(file_path) as file:
        for timestamp_str in file:
            timestamp_str = timestamp_str.rstrip()
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
            timestamps.append(timestamp)
    return timestamps


def parse_coordinates(file_path):
    """
    Parse csv file with list of coordinates (they may be in "x,y,z" or "x,z" format)

    :param file_path: Path where file is located
    :return: List of coordinates where each set of coordinates is a sub-list (e.g. [x,y,z])
    """
    coordinates = list()
    with open(file_path) as file:
        for line in file:
            parts = line.rstrip().split(",")
            x = float(parts[0])
            y = float(parts[1])
            z = None
            if len(parts) >= 3:
                z = float(parts[2])
            coordinates.append([x, y, z])
    return coordinates


def parse_mon_devices_info(file_path):
    """
    Parse csv file with monitoring devices coordinates into a dict structure, as shown next:
    mds_info = {
        "md_id1" : [x1,y1,z1],
        "md_id2" : [x2,y2,z2],
        ...
        "md_idN" : [xN,yN,zN]
    }

    :param file_path: Path where file is located
    :return: Dict with monitoring devices coordinates
    """
    mds_info = dict()
    with open(file_path) as file:
        for line in file:
            parts = line.rstrip().split(",")
            md_id = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            mds_info[md_id] = [x, y, z]

    sorted_dict = {key: value for key, value in sorted(mds_info.items())}
    mds_info = sorted_dict
    return mds_info


def parse_coordinates_info(file_path):
    """
    Parse csv file with the coordinates of reference points and their ids into a dict structure.

    coords_info = {
        "id_point1" : [x1,y1,z1],
        "id_point2" : [x2,y2,z2],
        ...
        "id_pointN" : [xN,yN,zN]
    }

    :param file_path: Path where file is located
    :return: Dict with reference point coordinates
    """
    coord_info = dict()
    with open(file_path) as file:
        for line in file:
            parts = line.rstrip().split(",")
            id_point = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = None
            if len(parts) >= 4:
                z = float(parts[3])
            coord_info[id_point] = [x, y, z]
    return coord_info


def parse_info_by_ap(file_path):
    """
    Parse csv file with RSSI information or channel information that is stored according to the AP's id, into a dict
    struture, e.g.:

    In case the file contains RSSI values:
    rssis = [
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint 1
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint 2
            ...
            {"ap0" : rssi, "ap1" : rssi, ...},   # RSSIs for fingerprint N
    ]

    In case the file contains channel values:
    channels = [
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint 1
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint 2
            ...
            {"ap0" : ch, "ap1" : ch, ...},   # Channels for fingerprint N
         ]

    :param file_path: Path where file is located
    :return: Dict with the RSSI or channel for each AP, using the AP's id as key
    """
    data_list = list()
    with open(file_path) as file:
        counter = 0
        for line in file:
            # print(line)
            info_fp = dict()
            info_aps = line.rstrip().split(",")
            for i in range(len(info_aps)):
                id_ap_info = info_aps[i].split(":")
                ap_id = id_ap_info[0]
                try:
                    info = int(id_ap_info[1])
                except IndexError:
                    print(file_path)
                    print("Error..." + str(i) + " SIZE: " + str(len(info_aps)))
                    print(line)
                    print("----------------")
                    print(info_aps)
                    print("----------------")
                    print(info_aps[i])
                    print("AP id: " + str(ap_id))
                    print("Line Counter: " + str(counter))
                    exit()
                info_fp[ap_id] = info
            data_list.append(info_fp)
            counter += 1
    return data_list


def get_aps_ids(dataset):
    """
    Find ids of APs in dataset.

    :param dataset: Dict with dataset information
    :return: List of detected APs ids
    """
    aps_ids = list()
    for i in range(len(dataset['rssis'])):
        for id_ap in dataset['rssis'][i].keys():
            if id_ap not in aps_ids:
                aps_ids.append(id_ap)
    return aps_ids


def get_positions(dataset):
    """
    Get list of unique positions that are within this dataset.

    :param dataset: Dict with dataset information
    :return: List of unique positions in the dataset
    """
    coords = list()

    for i in range(len(dataset['coordinates'])):
        found = False
        for point in coords:
            if point == dataset['coordinates'][i]:
                found = True
        if not found:
            coords.append(dataset['coordinates'][i].copy())

    return coords

