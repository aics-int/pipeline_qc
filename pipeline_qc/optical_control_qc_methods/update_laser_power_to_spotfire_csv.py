# How to use this script
# ...
folder_to_laser_power_csvs = r'\\allen\aics\microscopy\PRODUCTION\OpticalControl\ARGO-POWER\ZSD1\power_meter'
path_to_laser_power_csv = 'POWER_HM_power_meter_2021_01_05_at_16_03_45.csv'

date = "04/03/2019" # mm/dd/yyyy
system = "ZSD1" # ZSDx
your_initials = "CY" # xx
objective = "10X"

path_to_spotfire_csv = r'\\allen\aics\microscopy\Data\Instrumentation\power meter readings\laser_power_measurements_FinalForSpotfireVis.csv' # None to not write to spotfire csv, give a path to write to the spotfire csv


# Core script for running, no need to change
import os
import pandas as pd
from pipeline_qc.optical_control_qc_methods import laser_power_utils

# Transform Daybook csv to spotfire csv format
channels_power_dict = {'405': 280,
                       '488': 2300,
                       '561': 2400,
                       '638': 2400}

update_rows = laser_power_utils.process_argo_csv(
    csv_path=os.path.join(folder_to_laser_power_csvs, path_to_laser_power_csv),
    system=system,
    date=date,
    objective=objective,
    measurement_made_by_initial=your_initials,
    channels_power_dict=channels_power_dict
)

# If data (by date, system, objective, laser line, laser power) doesn't exist,
# update the spotfire csv

if path_to_spotfire_csv is not None:
    print('Updating spotfire csv')
    df = pd.read_csv(path_to_spotfire_csv)
    for index, row in update_rows.iterrows():
        if not (
                (df["Laser Line"] == row["Laser Line"]) &
                (df["% Laser Power"] == row["% Laser Power"]) &
                (df["Objective"] == row["Objective"]) &
                (df["Power at the objective (mW)"] == row["Power at the objective (mW)"]) &
                (df["System"] == row["System"]) &
                (df["Date"] == row["Date"])

        ).any():
            df = df.append(row)

    df.to_csv(path_to_spotfire_csv)
    print("Finished updating spotfire csv")