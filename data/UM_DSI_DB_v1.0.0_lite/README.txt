Copyright (c) 2022, University of Minho
The data are licensed under CC Attribution 4.0 International (CC BY 4.0)
The Python script files are provided under MIT license / X11 license.
This documentation is licensed under CC0 license.


Long-term database for UMinho's DSI building and supporting materials.
Version 1.0.0 2022.07.28

Directory content:

code/                   -> Python code to parse dataset and perform data analysis (create plots).
data/site_surveys/      -> Contains the measurements of the site surveys. Each subfolder contains the measurements of a data collection that occurred on a certain date. The name of the folder refers to the data collection. Each subfolder includes the following files: 
            - time.csv (timestamps of each Wi-Fi sample); 
            - coordinates.csv (cartesian coordinates of the location where each Wi-Fi sample was obtained); 
            - rssis.csv (signal strength values of detected APs in each Wi-Fi sample); 
            - channels.csv (frequency channels of each AP when the Wi-Fi sample was obtained).
data/mon_devices/		-> Contains the long-term measurements from monitoring devices. Each subfolder contains the Wi-Fi measurements of monitoring devices for each month considered, defined by the YYYY-MM format. Each month subfolder includes the same files as described in the previous point.
data/coords_info.csv    -> List of reference point coordinates and their names (rp_name,x,y). 
data/mds_info.csv       -> List of monitoring device coordinates and their names (md_name,x,y,z).



Citation request:

Ivo Silva, Cristiano Pendão, & Adriano Moreira. (2022). Continuous Long-term Wi-Fi Fingerprinting Dataset for Indoor Positioning (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6646008

Authors' contacts:
{ivo,cristiano,adriano}@dsi.uminho.pt

