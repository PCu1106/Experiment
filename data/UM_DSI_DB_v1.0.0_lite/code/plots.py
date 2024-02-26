import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from imageio import imread
import numpy as np
import os
from collections import Counter
import operator
import parser

running_mean_N = 72    # number of samples to average running mean (smooth the RSSI plot)
floor_plan_img = 'floor_plan.png'
absolute_path = os.path.abspath(__file__)


def running_mean(x, N):
    """
    Compute running mean of list, supports lists of any size.

    :param x: List to average
    :param N: Number of samples to consider in running mean
    :return: Averaged list
    """

    if len(x) <= 1:
        return x

    if N > len(x):
        N = len(x)-1

    smoothed_data = list()
    sum = 0.0
    # smoothing first elements of list
    for i in range(1, N):
        sum += x[i-1]
        val = sum/(i*1.0)
        smoothed_data.append(val)

    try:
        cumsum = np.cumsum(np.insert(x, 0, 0))
        smoothed_data.extend((cumsum[N:] - cumsum[:-N]) / float(N))
    except:
        print("An exception occurred")
        print("SIZE array: "+str(len(x)))
        print("SIZE smoothed_data: "+str(len(smoothed_data)))
        print("SIZE N: "+str(N))

    return smoothed_data


def get_date_formatter(timestamps):
    """
    Get the date formatter to set the x axis tick format in plots that have the time in the x axis

    :param timestamps: List of all timestamps to be considered (sorted in ascending order)
    :return: Date formatter
    """
    diff = timestamps[-1] - timestamps[0]
    df = None
    if diff.days < 1:
        df = mdates.DateFormatter('%H:%M\n%y/%m/%d')
    elif diff.days > 1 and diff.days<8:
        df = mdates.DateFormatter('%a %d/%m\n%Y')
    elif diff.days >= 8 and diff.days<60:
        df = mdates.DateFormatter('%d %b\n%Y')
    elif diff.days >= 60 and diff.days<180:
        df = mdates.DateFormatter('%Y-%m')
    elif diff.days >= 180:
        df = mdates.DateFormatter('%b\n%Y')

    return df


def set_plot_configs():
    """
    Set the default plot configs.
    This function is called at the beginning of each plot

    :return:
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    datafile = cbook.get_sample_data(os.path.dirname(absolute_path)+"/"+floor_plan_img)

    img = imread(datafile)
    ax.set_xlim([-55, 55])
    ax.set_ylim([-1, 21])
    ax.spines['bottom'].set_color('#d1d1d1')
    ax.spines['top'].set_color('#d1d1d1')
    ax.spines['right'].set_color('#d1d1d1')
    ax.spines['left'].set_color('#d1d1d1')
    ax.tick_params(axis='x', colors='#d1d1d1')
    ax.tick_params(axis='y', colors='#d1d1d1')

    ax.grid(True, alpha=0.01)
    plt.xticks(np.arange(-55, 55, step=2.5), fontsize=6)
    plt.yticks(np.arange(0, 20, step=1), fontsize=6)
    # the extent parameter is very important in order to have the image positioned in the right place
    im_handle = plt.imshow(img, zorder=0, extent=[-53.9, 53.9, -0.5, 18.85], alpha=1.0)


def simple_plot(output_folder):
    """
    Simple plot with axis and floor plan.

    :param output_folder: Folder where plot will be saved
    :return:
    """
    set_plot_configs()
    
    plt.savefig(output_folder/'simple_plot.pdf', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.close()
    plt.cla()
    plt.clf()


def coordinates(output_folder, dataset):
    """
    Scatter plot to show the points where there are fingerprints.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :return:
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    datafile = cbook.get_sample_data(os.path.dirname(absolute_path)+"/"+floor_plan_img)

    img = imread(datafile)
    ax.set_xlim([-55, 55])
    ax.set_ylim([-1, 21])
    ax.spines['bottom'].set_color('#d1d1d1')
    ax.spines['top'].set_color('#d1d1d1')
    ax.spines['right'].set_color('#d1d1d1')
    ax.spines['left'].set_color('#d1d1d1')
    ax.tick_params(axis='x', colors='#d1d1d1')
    ax.tick_params(axis='y', colors='#d1d1d1')

    ax.grid(True, alpha=0.01)
    # plt.axis('off') # hiding axis
    plt.xticks(np.arange(-55, 55, step=2.5), fontsize=6)
    plt.yticks(np.arange(0, 20, step=1), fontsize=6)

    # the extent parameter is very important in order to have the image positioned in the right place
    im_handle = plt.imshow(img, zorder=0, extent=[-53.9, 53.9, -0.5, 18.85], alpha=1.0)

    coords = dataset['coordinates']
    # count number of times each coordinate appears?
    x = list()
    y = list()
    [x.append(xy[0]) for xy in coords]
    [y.append(xy[1]) for xy in coords]

    plt.scatter(x, y, marker='D', color='gold', s=75, alpha=1, edgecolor='black', linewidth=0.8, label='Testing Point', zorder=1)
    plt.legend(loc='best')

    plt.savefig(str(output_folder)+'_coordinates.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()


def coordinates_density(output_folder, dataset):
    """
    Plot locations where Wi-Fi samples were collected and density of samples at each location.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :return:
    """
    set_plot_configs()

    coords = dataset['coordinates']
    coords_str = [str(coords[i][0])+","+str(coords[i][1]) for i in range(len(coords))]
    samples_per_point = Counter(coords_str)

    # count number of times each coordinate appears?
    x = list()
    y = list()
    occurrences = list()
    for point_str, number_occurrences in samples_per_point.items():
        x.append(float(point_str.split(",")[0]))
        y.append(float(point_str.split(",")[1]))
        occurrences.append(number_occurrences)

    index, occ_max = max(enumerate(occurrences), key=operator.itemgetter(1))

    # Plot data
    # Info about colormaps (cmap) https://matplotlib.org/tutorials/colors/colormaps.html
    color_map = "RdYlBu"
    sc = plt.scatter(x, y, marker='o', c=occurrences, vmax=occ_max, vmin=0, s=140, alpha=1, cmap=color_map, zorder=1)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.0%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)  # to adjust colorbar to the height of axis
    cbar.set_label('No. of Samples')

    plt.savefig(str(output_folder)+'_coordinates_density.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()


def mds_over_time(mds, output_folder, dataset):
    """
    Plot Monitoring Devices over time, showing periods during which each Monitoring Device was operational and
    collecting data.

    :param mds: Monitoring Devices info
    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :return:
    """
    plt.figure(figsize=(6.5, 4))

    # For each monitoring device
    # get timestamps when it was detected
    count_md = 0
    for md_id, pos in mds.items():
        count_md += 1
        # list of timestamps when MD collected a Wi-Fi sample
        md_timestamps = list()
        for i in range(len(dataset['timestamps'])):

            if dataset['coordinates'][i][0] == pos[0] and dataset['coordinates'][i][1] == pos[1]:
                md_timestamps.append(dataset['timestamps'][i])

        # plot for this monitoring device
        plt.scatter(md_timestamps, [count_md] * len(md_timestamps), marker='|', label=md_id, rasterized=True)

    fontsize = 7
    y_values = mds.keys()
    y_axis = np.arange(1, len(y_values)+1, 1)

    plt.xticks(fontsize=fontsize) # rotation='vertical',
    plt.yticks(y_axis, y_values, fontsize=fontsize)

    plt.gca().xaxis.set_major_formatter(get_date_formatter(dataset['timestamps']))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16, min_n_ticks=14))  # set number of ticks
    plt.gcf().autofmt_xdate(rotation=0, ha='center')

    # make sure plot uses all space
    plt.ylim(0, len(y_values)+1)
    plt.xlim(dataset['timestamps'][0], dataset['timestamps'][-1])

    plt.savefig(str(output_folder)+'mds_vs_time.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()


def aps_over_time(output_folder, dataset, type='global'):
    """
    Processing to obtain the data that will be plotted in the APs over time plot.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :param type: 'global' or 'local';
                'global' when fingerprints from all reference points are considered;
                'local' only fingerprints of a reference point are considered.
    :return:
    """

    # Sort by first date when AP is detected
    # first obtain all aps ids

    if type == 'global':
        aps_ids = list()
        aps_times = dict()
        aps_rssis = dict()

        for i in range(len(dataset['timestamps'])):
            for id_ap, rssi in dataset['rssis'][i].items():
                if id_ap not in aps_times:
                    aps_ids.append(id_ap)
                    aps_times[id_ap] = list()
                    aps_rssis[id_ap] = list()
                aps_times[id_ap].append(dataset['timestamps'][i])
                aps_rssis[id_ap].append(dataset['rssis'][i][id_ap])
        filename = str(output_folder)+'aps_vs_time.pdf'
        plot_aps_over_time(aps_ids, aps_times, aps_rssis, dataset, filename)

    elif type == 'local':
        positions = parser.get_positions(dataset)
        for pos in positions:
            x = pos[0]
            y = pos[1]
            aps_ids = list()
            aps_times = dict()
            aps_rssis = dict()

            for i in range(len(dataset['timestamps'])):
                if dataset['coordinates'][i][0] == x and dataset['coordinates'][i][1] == y:
                    for id_ap, rssi in dataset['rssis'][i].items():
                        if id_ap not in aps_times:
                            aps_ids.append(id_ap)
                            aps_times[id_ap] = list()
                            aps_rssis[id_ap] = list()
                        aps_times[id_ap].append(dataset['timestamps'][i])
                        aps_rssis[id_ap].append(dataset['rssis'][i][id_ap])
            filename = str(output_folder)+'aps_vs_time'+'_'+'x{0:.2f}'.format(x)+'_y{0:.2f}'.format(y)+'.pdf'
            # create plot for each position
            plot_aps_over_time(aps_ids, aps_times, aps_rssis, dataset, filename)


def plot_aps_over_time(aps_ids, aps_times, aps_rssis, dataset, filename):
    """
    Generate the plot with APs that were detected over time.
    Each coloured line represents a specific AP during the periods when it was detected.

    :param aps_ids: List of APs' ids
    :param aps_times: Dict that has the list of times when each AP was detected (id_ap is the key)
    :param aps_rssis: Dict that has the list of RSSIs when each AP was detected (id_ap is the key)
    :param dataset: Dict with dataset information
    :param filename: Path of the file that will be saved, including filename.
    :return:
    """

    plt.figure(figsize=(6.5, 4))
    linewidth = 0.35
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

    # Filter out APs that are rarely seen based on percentage of times APs that are detected
    # over the number of fingerprints
    ap_detection_rate_th = 0.0001     # used to ignore APs that are rarely detected (5% of fingerprints)

    ap_count = 0
    for id_ap in aps_ids:
        if (len(aps_times[id_ap])*1.0) / (len(dataset['timestamps'])*1.0) > ap_detection_rate_th:
            ap_on_off = [ap_count if x is not None else x for x in aps_rssis[id_ap]]
            plt.plot(aps_times[id_ap], ap_on_off, '-', linewidth=linewidth, alpha=1.0, rasterized=True)
            ap_count += 1

    axis_color = "#000000"      # "#000000" <- Black      #"#d1d1d1" <- light gray
    fontsize = 5
    plt.xticks(fontsize=fontsize)
    plt.ylabel(r"$AP_{0 ... N} $", fontsize=fontsize, color=axis_color) # r'\textbf{time} (s)'        r"$\displaystyle $"
    plt.yticks(np.arange(0, ap_count+10, 40.0), fontsize=fontsize)
    plt.grid(alpha=0.3, linestyle='--', axis='x')

    plt.gca().xaxis.set_major_formatter(get_date_formatter(dataset['timestamps']))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16, min_n_ticks=14))  # set number of ticks
    plt.gcf().autofmt_xdate(rotation=0, ha='center')
    #plt.setp(plt.xticks()[1], fontsize=fontsize)    #rotation=30,ha='right'

    # make sure plot uses all space
    plt.ylim(-1, ap_count+10)
    plt.xlim(dataset['timestamps'][0], dataset['timestamps'][-1])

    ax = plt.gca()

    # setting all axes and ticks to grey
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['top'].set_color(axis_color)
    ax.spines['right'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)

    print("Total plotted APs: "+str(ap_count))

    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()


def ap_mean_rssi(output_folder, dataset, id_ap):
    """
    Plot the average RSSI value measured at each position for a given AP ID.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :param id_ap: ID of the AP that will be plotted
    :return:
    """

    # Get set of unique positions in the dataset
    positions = parser.get_positions(dataset)
    x, y, mean_rssis = list(), list(), list()

    for pos in positions:
        # keep a list of rssi values of this AP at this position
        rssis = list()
        for i in range(len(dataset['coordinates'])):
            # the rssi is considered if it is detected at this position
            if pos == dataset['coordinates'][i] and id_ap in dataset['rssis'][i].keys():
                rssis.append(dataset['rssis'][i][id_ap])

        if len(rssis) > 0:
            mean_rssi = np.mean(rssis)
        else:
            mean_rssi = -120 # default value, because AP was not detected

        x.append(pos[0])
        y.append(pos[1])
        mean_rssis.append(mean_rssi)

    set_plot_configs()

    # Plot data
    # Info about colormaps (cmap) https://matplotlib.org/tutorials/colors/colormaps.html
    color_map = "Spectral"  #"RdYlGn"
    sc = plt.scatter(x, y, marker='o', c=mean_rssis, vmax=-10, vmin=-120, s=120, alpha=1, cmap=color_map, zorder=1)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.0%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)  # to adjust colorbar to the height of axis
    cbar.set_label('Mean RSSI (dBm)')

    plt.savefig(str(output_folder)+'mean_rssi_'+str(id_ap)+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()
    return


def ap_rssi_over_time(output_folder, dataset, id_ap, position):
    """
    Plot RSSI of AP over time for a specific reference point of the dataset.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :param id_ap: ID of the AP
    :param position: list with two items [x,y] that identify the location where samples were obtained
    :return:
    """
    times = list()
    rssis = list()
    x = position[0]
    y = position[1]
    for i in range(len(dataset['timestamps'])):

        # Make sure that this AP was observed in this position
        if dataset['coordinates'][i][0] == x and dataset['coordinates'][i][1] == y:
            if id_ap in dataset['rssis'][i]:
                times.append(dataset['timestamps'][i])
                rssis.append(dataset['rssis'][i][id_ap])


    plt.figure(figsize=(6.5, 4))
    fontsize = 6
    plt.xticks(fontsize=fontsize)
    plt.ylabel("RSSI (dBm)")
    plt.grid(alpha=0.3, linestyle='-')
    plt.scatter(times, rssis, marker='.', edgecolors='none', color='C0', label='Raw', rasterized=True, zorder=2)
    plt.plot(times, running_mean(rssis, N=running_mean_N), color='C1', label='Smoothed', rasterized=True, zorder=4)

    plt.gca().xaxis.set_major_formatter(get_date_formatter(dataset['timestamps']))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16, min_n_ticks=14))  # set number of ticks
    plt.gcf().autofmt_xdate(rotation=0, ha='center')
    plt.yticks(np.arange(-110, -10, 20.0), fontsize=6)
    plt.ylim(-110, -10)
    plt.xlim(dataset['timestamps'][0], dataset['timestamps'][-1])
    plt.legend()
    plt.savefig(str(output_folder)+'rssi_vs_time_'+str(id_ap)+'_'+'x{0:.2f}'.format(x)+'_y{0:.2f}'.format(y)+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()

    return


def ap_rssi_over_time_global(output_folder, dataset, id_ap, mds_info):
    """
    Generate plots of the AP's RSSI over time for each reference point of the dataset.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :param id_ap: ID of the AP
    :return:
    """

    plt.figure(figsize=(6.5, 4))
    fontsize = 6
    plt.xticks(fontsize=fontsize)
    plt.ylabel("RSSI (dBm)")
    plt.grid(alpha=0.3, linestyle='-')

    # for each MD - plot the mean rssi
    count = 0
    for md_id, pos in mds_info.items():
        times = list()
        rssis = list()
        for i in range(len(dataset['timestamps'])):
            # Make sure that this AP was observed in this position
            if dataset['coordinates'][i][0] == pos[0] and dataset['coordinates'][i][1] == pos[1]:
                if id_ap in dataset['rssis'][i]:
                    times.append(dataset['timestamps'][i])
                    rssis.append(dataset['rssis'][i][id_ap])
        plt.plot(times, running_mean(rssis, N=running_mean_N), color='C'+str(count), label=md_id+'', rasterized=True, zorder=3)
        count += 1

    plt.gca().xaxis.set_major_formatter(get_date_formatter(dataset['timestamps']))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16, min_n_ticks=14))  # set number of ticks
    plt.gcf().autofmt_xdate(rotation=0, ha='center')
    plt.yticks(np.arange(-110, -10, 20.0), fontsize=6)
    plt.ylim(-110, -10)
    plt.xlim(dataset['timestamps'][0], dataset['timestamps'][-1])
    plt.legend(fontsize=4)
    plt.savefig(str(output_folder)+'rssi_vs_time_'+str(id_ap)+'_global.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()

    return


def ap_channel_over_time(output_folder, dataset, id_ap, position=None, type='global'):
    """
    Processing to obtain the data that will be plotted in the APs over time plot.

    :param output_folder: Folder where plot will be saved
    :param dataset: Dict with dataset information
    :param id_ap: ID of the AP
    :param position: list with two items [x,y] that identify the location where samples were obtained (can be 'None')
    :param type: 'global' or 'local';
                'global' when fingerprints from all reference points are considered;
                'local' only fingerprints of a reference point are considered.
    :return:
    """
    times = list()
    channels = list()

    if type == 'local' and position is not None:
        for i in range(len(dataset['timestamps'])):
            # Make sure that this AP was observed in this position
            if dataset['coordinates'][i][0] == position[0] and dataset['coordinates'][i][1] == position[1]:
                if id_ap in dataset['channels'][i]:
                    times.append(dataset['timestamps'][i])
                    channels.append(dataset['channels'][i][id_ap])
        filename = str(output_folder)+'channel_vs_time_'+str(id_ap)+'_'+'x{0:.2f}'.format(position[0])+'_y{0:.2f}'.format(position[1])+'.pdf'

    elif type == 'global':
        for i in range(len(dataset['timestamps'])):
            if id_ap in dataset['channels'][i]:
                times.append(dataset['timestamps'][i])
                channels.append(dataset['channels'][i][id_ap])
        filename = str(output_folder)+'channel_vs_time_'+str(id_ap)+'_global.pdf'

    plot_channel_over_time(times, channels, dataset, id_ap, filename)
    return


def plot_channel_over_time(times, channels, dataset, id_ap, filename):
    """
    Plot AP's channel over time.

    :param times: List of timestamp values.
    :param channels: List of channel values.
    :param dataset: Dict with dataset information
    :param id_ap: ID of the AP
    :param filename: Path of the file that will be saved, including filename.
    :return:
    """

    # Detect whether this AP is emitting 2.4GHz Channels or 5GHz channels
    ch24ghz = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ch5ghz = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165]
    ap_freq = "NONE"

    if any(elem in ch24ghz for elem in channels):
        ap_freq = "2.4GHz"
    elif any(elem in ch5ghz for elem in channels):
        ap_freq = "5GHz"

    plt.figure(figsize=(6.5, 4))
    fontsize = 6
    plt.xticks(fontsize=fontsize-2)
    plt.ylabel("Channel")
    plt.grid(alpha=0.3, linestyle='-')

    plt.scatter(times, channels, marker='.', s=12, edgecolors='black', linewidth=0.2, label=id_ap, rasterized=True, zorder=3)

    plt.gca().xaxis.set_major_formatter(get_date_formatter(dataset['timestamps']))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16, min_n_ticks=14))  # set number of ticks
    plt.gcf().autofmt_xdate(rotation=0, ha='center')

    if ap_freq == "2.4GHz":
        plt.yticks(np.arange(-1, 14, 1.0), fontsize=fontsize)
    elif ap_freq == "5GHz":
        plt.yticks(ch5ghz, ch5ghz, fontsize=fontsize)

    plt.xlim(dataset['timestamps'][0], dataset['timestamps'][-1])
    plt.legend()
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.cla()
    plt.clf()
    return

