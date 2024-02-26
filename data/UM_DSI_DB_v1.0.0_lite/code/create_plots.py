import parser
import sys
import os
import platform
import plots
from pathlib import Path

"""

Main script that parses the dataset and generates the plots

"""


# Set these flags to true or false to enable/disable plotting
# of site surveys data or the monitoring devices data
plot_site_surveys_data = True
plot_mon_dev_data = True

# Due to the existence of many APs, only a few APs are plotted
# set this value in order to define the number of APs that are plotted
n_aps_to_plot = 20

plots_folder = ""
ssv_folder = ""
mds_folder = ""
mds_info_path = ""


def print_args_error():
    """
    Show error when the arguments are not properly set
    :return:
    """
    print("Please set the arguments when running the script. Make sure the paths are correct. Example usage:")
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        print("python3 create_plots.py [path/to/main/dataset/folder] [path/to/folder/for/generated/plots/]")
    elif platform.system() == 'Windows':
        print("python3 create_plots.py [C:\\path\\to\\main\\dataset\\folder] [C:\\path\\for\\generated\\plots]")


def parse_args(argv):
    """
    Handle the input arguments to set the paths used to parse the dataset and generate plots

    The command to run the script has the following format:

        $ python3 script_name.py arg1 arg2

    Where arg1 represents the directory main folder of the dataset which should include the 'data' sub-folder; and
    arg2 is the directory where plots are saved into

    :param argv: list of parsed arguments
    :return:
    """

    print(argv)
    global plots_folder
    global ssv_folder
    global mds_folder
    global mds_info_path

    if len(argv) == 2:

        dataset_folder = Path(argv[0])
        plots_folder = Path(argv[1])
        if os.path.exists(dataset_folder) and os.path.exists(plots_folder):
            ssv_folder = dataset_folder/'data'/'site_surveys'
            mds_folder = dataset_folder/'data'/'mon_devices'
            mds_info_path = dataset_folder/'data'/'mds_info.csv'

            if ssv_folder.is_dir() and mds_folder.is_dir() and mds_info_path.is_file() and plots_folder.is_dir():
                print("Loaded the following directories:")
                print(ssv_folder)
                print(mds_folder)
                print(mds_info_path)
            else:
                print("Please make sure that the dataset has all subfolders inside.")
                sys.exit()
        else:
            print_args_error()
            sys.exit()
    else:
        print_args_error()
        sys.exit()


if __name__ == '__main__':

    # Handle arguments
    parse_args(sys.argv[1:])
    print(ssv_folder)
    site_surveys = [f.name for f in os.scandir(ssv_folder) if f.is_dir()]
    site_surveys = parser.sorted_nicely(site_surveys)
    print("List of site surveys:")
    print(site_surveys)

    mds_months = [f.name for f in os.scandir(mds_folder) if f.is_dir()]
    # sort folders in ascending order
    mds_months = parser.sorted_nicely(mds_months)

    # Basic plot to show floor plan
    plots.simple_plot(plots_folder)
    if plot_site_surveys_data:
        # Create target Directory if it doesn't exist
        if not os.path.exists(plots_folder/"site_surveys"):
            os.mkdir(plots_folder/"site_surveys")
            print("Directory ", plots_folder/"site_surveys",  " Created ")

        for site_survey in site_surveys:
            print("Parsing site survey: "+str(site_survey))
            dataset = parser.parse_dataset(ssv_folder/site_survey)
            plots.coordinates(plots_folder/"site_surveys"/site_survey, dataset)
            plots.coordinates_density(plots_folder/"site_surveys"/site_survey, dataset)
            aps_ids = parser.get_aps_ids(dataset)
            coords = parser.get_positions(dataset)

            # Plot APs rssi of a limited number of APs
            for i in range(n_aps_to_plot):
                path_str = str(plots_folder/"site_surveys"/site_survey)
                plots.ap_mean_rssi(path_str+"_", dataset, aps_ids[i])

                # RSSI and channel over time are only plotted for the first reference point (coords[0])
                plots.ap_rssi_over_time(path_str+"_", dataset, aps_ids[i], position=coords[0])
                plots.ap_channel_over_time(path_str+"_", dataset, aps_ids[i], position=coords[0])
    else:
        print("Plotting of SITE-SURVEYS is disabled.")
        print("To enabled it, set this value: plot_site_surveys_data=True")


    if plot_mon_dev_data:
        # Create target Directory if it doesn't exist
        if not os.path.exists(plots_folder/"mon_devices"):
            os.mkdir(plots_folder/"mon_devices")
            print("Directory ", plots_folder/"mon_devices/",  " Created ")

        mds = parser.parse_mon_devices_info(mds_info_path)

        # Parsing month datasets into one large dataset
        long_term_dataset = dict()
        long_term_dataset['timestamps'] = list()
        long_term_dataset['coordinates'] = list()
        long_term_dataset['rssis'] = list()
        long_term_dataset['channels'] = list()
        print("Parsing long-term datasets:")
        for mds_month in mds_months:
            print(mds_month)
            dataset = parser.parse_dataset(mds_folder/mds_month)
            long_term_dataset['timestamps'].extend(dataset['timestamps'])
            long_term_dataset['coordinates'].extend(dataset['coordinates'])
            long_term_dataset['rssis'].extend(dataset['rssis'])
            long_term_dataset['channels'].extend(dataset['channels'])

            plots.coordinates_density(plots_folder/"mon_devices"/mds_month, dataset)
            aps_ids = parser.get_aps_ids(long_term_dataset)
            positions = parser.get_positions(long_term_dataset)
            plots_folder_mds_month = str(plots_folder/"mon_devices"/mds_month)+"/"
            
            ## PLOTS MONTHLY DATA ##
            count = 0
            for id_ap in aps_ids:
                if count < n_aps_to_plot:
                    # Plot AP data
                    
                    # Create target Directory if it doesn't exist
                    if not os.path.exists(plots_folder_mds_month):
                        os.mkdir(plots_folder_mds_month)
                        print("Directory ", plots_folder_mds_month,  " Created ")
                    plots.ap_channel_over_time(plots_folder_mds_month, dataset, id_ap, position=None, type='global')
                    plots.ap_rssi_over_time_global(plots_folder_mds_month, dataset, id_ap, mds)
                    # Plot AP rssi over time for each of the monitoring devices
                    for md_id in mds.keys():
                        pos_x = mds[md_id][0]
                        pos_y = mds[md_id][1]
                        plots.ap_rssi_over_time(plots_folder_mds_month, dataset, id_ap, position=[pos_x, pos_y])
                        plots.ap_channel_over_time(plots_folder_mds_month, dataset, id_ap, position=[pos_x, pos_y], type='local')
                else:
                    break
                count += 1

        aps_ids = parser.get_aps_ids(long_term_dataset)
        positions = parser.get_positions(long_term_dataset)
        ## PLOTS LONG-TERM DATA ##
        count = 0
        plots_folder_mds = str(plots_folder/"mon_devices")+"/"
        for id_ap in aps_ids:
            if count < n_aps_to_plot:
                # Plot AP data
                plots.ap_channel_over_time(plots_folder_mds, long_term_dataset, id_ap, position=None, type='global')
                plots.ap_rssi_over_time_global(plots_folder_mds, long_term_dataset, id_ap, mds)

                # Plot AP rssi over time for each of the monitoring devices
                for md_id in mds.keys():
                    pos_x = mds[md_id][0]
                    pos_y = mds[md_id][1]
                    plots.ap_rssi_over_time(plots_folder_mds, long_term_dataset, id_ap, position=[pos_x, pos_y])
                    plots.ap_channel_over_time(plots_folder_mds, long_term_dataset, id_ap, position=[pos_x, pos_y], type='local')
            else:
                break
            count += 1

        print("Long-term Dataset Info:")
        print("timestamps: %d" % len(long_term_dataset['timestamps']))
        print("coordinates: %d" % len(long_term_dataset['coordinates']))
        print("rssis: %d" % len(long_term_dataset['rssis']))
        print("channels: %d" % len(long_term_dataset['channels']))
        print("Number of Detected APs: %d" % len(aps_ids))

        # Plot Monitoring Devices over time (periods during which they were active collecting data)
        print("Plot MDs vs time.")
        plots.mds_over_time(mds, plots_folder_mds, long_term_dataset)

        # globally: APs visible by any monitoring device over time
        print("Plot detected APs vs time.")
        plots.aps_over_time(plots_folder_mds, long_term_dataset, type='global')

        # locally: APs visible by monitoring device over time
        plots.aps_over_time(plots_folder_mds, long_term_dataset, type='local')

        plots.coordinates_density(plots_folder_mds+"global_", long_term_dataset)

    else:
        print("Plotting of Monitoring Devices data is disabled.")
        print("To enabled it, set this value: plot_mon_dev_data=True")

