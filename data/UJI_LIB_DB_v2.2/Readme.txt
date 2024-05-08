Copyright (c) 2018, Universtitat Jaume I (UJI)
The data is licensed under CC Attribution 4.0 International (CC BY 4.0).
The script files are provided under MIT license / X11 license.
This documentation is licensed under CC0 license.

WiFi measurements database for UJI's library and supporting material.
Version 2.0 16-08-2018

Directory content:

./      ->      Common utility functions and sample usage Matlab scripts. Additionally, nearAPs.csv identifies the APs (columns in the database) that correspond to the 4 wireless devices closest to the collection area and provides the device position and the emitted AP's characteristics. 
db/     ->      Database measurements. Within it, each subfolder contains the measurements
				of each dataset that belongs to a collection month. Each dataset is represented
				by four files: the RSS, the time, the coordinates, and the identifiers files.
files/  ->      Matlab functions for loading the measurements.
ids/    ->      Matlab functions for filtering loaded samples.
ips/	->      Matlab functions provinding sample IPS methods.
shelves/ ->     Bookshelves coordinates and related Matlab scripts.


Citation request: 

Mendoza-Silva, G.M.; Richter, P.; Torres-Sospedra, J.; Lohan, E.S.; Huerta, J. Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning. Data 2018, 3, 3.

G.M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E.S. Lohan, J. Huerta, A. Cramariuc, "Long-Term Wi-Fi fingerprinting dataset and supporting material", Zenodo repository, DOI 10.5281/zenodo.1066041.

gmendoza@uji.es
philipp.richter@tut.fi
jtorres@uji.es
elena-simona.lohan@tut.fi
huerta@uji.es
