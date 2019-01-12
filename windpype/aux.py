# coding=utf-8
### 
### Submodule aux
###

print('windpype submodule "aux" imported')

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from argparse import Namespace

### Data handling related

def fill_missing_values(data_df,time_step=60*60):
    """ Data handling: Function that fills in missing time steps with the most recent value for all columns of data. 

    Parameters
    ----------
    data_df: object
        Pandas dataframe containing the timeseries.
    time_step: int/float
        Desired time step, default: 60*60 seconds

    """

    print('Check min max of datetime')
    print(min(data_df['datetime']))
    print(max(data_df['datetime']))

    i_missing = np.where(data_df['time_steps'].values != time_step)[0]

    new_delta_time = np.arange(0,max(data_df['delta_time'].values)+1e-6,time_step)
    old_delta_time = data_df['delta_time'].values
    i_existing = np.array([_ in old_delta_time for _ in new_delta_time])
    i_missing = np.where(i_existing == False)[0]
    i_existing = np.where(i_existing == True)[0]

    # Create new complete dataframe
    new_time_steps = np.zeros(len(new_delta_time))+time_step
    new_data_df = pd.DataFrame({'delta_time':new_delta_time,'time_steps':new_time_steps}).reset_index(drop=True)
    new_len = len(new_data_df)

    for key in data_df.keys():
        if key != 'delta_time':
            new_array = np.zeros(new_len)
            new_array[i_existing] = data_df[key]
            for i in i_missing:
                new_array[i] = new_array[i-1]
            new_data_df[key] = new_array

    new_data_df['datetime'] = pd.date_range(start=min(data_df['datetime']), periods=len(new_data_df), freq='%ss' % time_step)
    print('Check min max of datetime')
    print(min(new_data_df['datetime']))
    print(max(new_data_df['datetime']))
    new_data_df = update_times(new_data_df)

    print('Length of old dataframe: %s' % (len(data_df)))
    print('Length of new dataframe: %s' % (len(new_delta_time)))

    return(new_data_df)

def CombineDataFrames(data_df_1,data_df_2,cut=False,**kwargs):
    """ Data handling: Function that combines two dataframes into one. 
    
    Parameters
    ----------
    data_df_1: dataframe
        Dataframe 1, should be of the finest time resolution
    data_df_2: dataframe
        Dataframe 2
    method: str
        Which method to use for combining the dataframes, default: 'append'
        - 'append': add data_df_2 onto data_df_1
        - 'merge': merge the two together, on time stamps
        - 'stitch': append the latter on the first dataframe, cutting away overlapping time stamps from the latter
    """

    args                =   dict(method='merge')
    args                =   update_dictionary(args,kwargs)
    method              =   args['method']
    print('Using method: %s' % method)

    len_df_1 = len(data_df_1)
    len_df_2 = len(data_df_2)

    print('Length of dataframe 1: %s' % len_df_1)
    print('Length of dataframe 2: %s' % len_df_2)

    if method == 'append':
        combined_data_df = data_df_1.append(data_df_2,ignore_index=True)

    if method == 'merge':
        # This will put nans where a column is missing a row element...
        combined_data_df = pd.merge(data_df_1,data_df_2,on='datetime',how='outer')

        # Set all nans to previous value
        combined_data_df = combined_data_df.sort_values('datetime').reset_index(drop=True)
        combined_data_df = combined_data_df.fillna(method='ffill')
        # Set any remaining nans to -1
        combined_data_df = combined_data_df.fillna(-1)

        # Check for duplicate columns
        for key in combined_data_df.keys():
            if '_x' in key:
                combined_data_df[key.replace('_x','')] = combined_data_df[key].values
                combined_data_df = combined_data_df.drop(key,axis=1)
                combined_data_df = combined_data_df.drop(key.replace('_x','')+'_y',axis=1)
        combined_data_df = combined_data_df.sort_values('datetime')

    if method == 'stitch':
        # Find which dataframe comes first (later timestamps should overwrite earlier ones because of missing data):
        if min(data_df_2['datetime']) > min(data_df_1['datetime']): first_df,second_df = data_df_1,data_df_2
        if min(data_df_2['datetime']) < min(data_df_1['datetime']): first_df,second_df = data_df_2,data_df_1
        max_time_1 = max(first_df['datetime'])
        min_time_2 = min(second_df['datetime'])
        if max_time_1 > min_time_2:
            mask = np.array([first_df['datetime'] < min_time_2])[0]
            first_df = first_df[mask] # cut out last bit of earliest data set (usually ends in no data...)
        else:
            print('Gap between data sets!')
            print(max_time_1,min_time_2)
            a = asfa 
        combined_data_df = first_df.append(second_df,ignore_index=True)

    combined_data_df = update_times(combined_data_df)

    min_datetime = np.min(combined_data_df['datetime'])
    max_datetime = np.max(combined_data_df['datetime'])
    print('Combined data over common time period from %s to %s' % (min_datetime,max_datetime))
    print('%s datapoints' % len(combined_data_df))

    return(combined_data_df)

def SaveData(data_df,file_path='',file_name=''):
    """ Data handling: Function that saves dataframe. 

    Parameters
    ----------
    data_df: Pandas dataframe
        Dataframe to be stored.
    filepath: str
        default: ''
    file_name: str
        default: ''
    """

    data_df.to_pickle(file_path + file_name)

def update_times(data_df):
    """ Data handling: Function that updates date and times in dataframe.

    Parameters
    ----------
    data_df: Pandas dataframe
        Dataframe to updated. Important that it contains all datetimes stamps intact.

    """
    data_df = data_df.sort_values('datetime').reset_index(drop=True)
    datetimes = data_df['datetime']
    delta_times =  datetimes - min(datetimes)
    delta_times = [delta_time.total_seconds() for delta_time in delta_times]
    data_df['delta_time'] = delta_times
    time_steps = np.array([delta_times[i+1]-delta_times[i] for i in range(len(data_df)-1)])
    time_steps = np.append(time_steps[0],time_steps)
    data_df['time_steps'] = time_steps #s

    return(data_df)

def get_STD(time,values,time_cut=False):

    if time_cut:
        mask = np.array([(time >= time_cut[0]) & (time <= time_cut[1])])[0]
        values = values[mask]    
    try:
        STD = np.std(values[values != 0])
    except:
        STD = 0.
    # if np.isnan(STD):
    #     STD = 0.

    return(STD)

def get_percentile(time,values,percent,time_cut=False):

    if time_cut:
        mask = np.array([(time >= time_cut[0]) & (time <= time_cut[1])])[0]
        values = values[mask]    
    try:
        perc = np.percentile(values[values != 0], percent, axis=0)
    except:
        perc = 0.

    return(perc)

### Other

def identify_storms():
    """ Uses wikipedia list (https://da.wikipedia.org/wiki/Navngivne_storme_i_Danmark) to search for big storms.
    """

    archive = {\
        'Knud':[np.datetime64('2018-09-21 00:00:00'),np.datetime64('2018-09-22 00:00:00')],\
        'Johanne':[np.datetime64('2018-08-10 00:00:00'),np.datetime64('2018-08-11 00:00:00')],\
        'Ingolf':[np.datetime64('2017-10-29 00:00:00'),np.datetime64('2017-10-30 00:00:00')],\
        'Urd':[np.datetime64('2016-12-26 00:00:00'),np.datetime64('2016-12-27 00:00:00')],\
        'Helga':[np.datetime64('2015-12-04 00:00:00'),np.datetime64('2015-12-05 00:00:00')],\
        'Gorm':[np.datetime64('2015-11-29 00:00:00'),np.datetime64('2015-11-30 00:00:00')],\
        'Freja':[np.datetime64('2015-11-07 00:00:00'),np.datetime64('2015-11-08 00:00:00')],\
        'Egon':[np.datetime64('2015-01-10 00:00:00'),np.datetime64('2015-01-11 00:00:00')],\
        'Dagmar':[np.datetime64('2015-01-09 00:00:00'),np.datetime64('2015-01-10 00:00:00')],\
        'Alexander':[np.datetime64('2014-12-12 00:00:00'),np.datetime64('2014-12-13 00:00:00')],\
        'Carl':[np.datetime64('2014-03-14 00:00:00'),np.datetime64('2014-03-15 00:00:00')],\
        'Bodil':[np.datetime64('2013-12-05 00:00:00'),np.datetime64('2013-12-07 00:00:00')],\
        'Allan':[np.datetime64('2013-10-28 00:00:00'),np.datetime64('2013-10-29 00:00:00')]}

    datetime_storms = [\
        np.datetime64('2018-09-21 12:00:00'),\
        np.datetime64('2018-08-10 12:00:00'),\
        np.datetime64('2017-10-29 12:00:00'),\
        np.datetime64('2016-12-26 12:00:00'),\
        # np.datetime64('2016-10-30 12:00:00'),\
        np.datetime64('2015-12-04 12:00:00'),\
        np.datetime64('2015-11-29 12:00:00'),\
        np.datetime64('2015-11-07 12:00:00'),\
        np.datetime64('2015-01-10 12:00:00'),\
        np.datetime64('2015-01-09 12:00:00'),\
        np.datetime64('2014-12-12 12:00:00'),\
        np.datetime64('2014-03-14 12:00:00'),\
        np.datetime64('2013-12-06 00:00:00'),\
        np.datetime64('2013-10-28 12:00:00')]

    # OBS: I invented 2016 October 30 storm...

    # indices = np.array([])
    # for _,datetime in enumerate(datetimes):
    #     for key in archive.keys():
    #         datetime_range = archive[key]
    #         if (datetime > datetime_range[0]) & (datetime < datetime_range[1]):
    #             print('found match: %s, timestamp: %s' % (key,datetime))
    #             indices = np.append(indices,_)

    return(datetime_storms)

def zone(col_name):

    zone = ''

    if col_name == 'TotalWindPower' : zone = 'DK'
    if col_name == 'TotalWindPower_DK1' : zone = 'DK1'
    if col_name == 'TotalWindPower_DK2' : zone = 'DK2'
    if col_name == 'TotalWindPower_BO' : zone = 'BO'
    if col_name == 'OnshoreWindPower' : zone = 'onshore'
    if col_name == 'OffshoreWindPower' : zone = 'offshore'

    return(zone)

def update_dictionary(old_dict,new_values):
    """ Function that updates a dictionary

    Parameters
    ----------
    old_dict: dict
        Old dictionary
    new_values:
        Dictionary with the new entries to old_dict
    """

    for key in old_dict:
        if key in new_values:
            old_dict[key]     =   new_values[key]
    return old_dict

def pretty_label(name,percent=False):
    """ Creates a pretty label for plotting etc. for given column name.
    """

    label = name

    if name == 'TotalWindPower': label = 'All Wind Power in DK' 
    if name == 'SolarPower': label = 'Solar Power in DK' 
    if name == 'TotalWindPower_DK1': label = 'Wind Power in DK1' 
    if name == 'TotalWindPower_DK2': label = 'Wind Power in DK2' 
    if name == 'TotalWindPower_BO': label = 'Wind Power in Bornholm' 
    if name == 'accum_capacity_TotalWindPower_DK': label = 'Installed capacity in DK' 
    if name == 'accum_capacity_TotalWindPower_DK1': label = 'Installed capacity in DK1' 
    if name == 'accum_capacity_TotalWindPower_DK2': label = 'Installed capacity in DK2' 
    if name == 'accum_capacity_TotalWindPower_BO': label = 'Installed capacity in Bornholm' 
    if name == 'BalancingPowerPriceUpEUR_DK1': label = 'Balancing price up in DK1 [EUR]' 
    if name == 'BalancingPowerPriceUpEUR_DK2': label = 'Balancing price up in DK2 [EUR]' 
    if name == 'BalancingPowerPriceDownEUR_DK1': label = 'Balancing price down in DK1 [EUR]' 
    if name == 'BalancingPowerPriceDownEUR_DK2': label = 'Balancing price down in DK2 [EUR]' 
    if name == 'SpotPriceEUR': 
        label = 'Spot price [EUR]'
    if name == 'ResidualPrice_DK1': 
        label = 'Residual price in DK1' 
        if percent: label = label + ' [% of spot price]'
    if name == 'ResidualPrice_DK2': 
        label = 'Residual price in DK2' 
        if percent: label = label + ' [% of spot price]'
    if name == 'OnshoreWindPower': label = 'Onshore Wind Power in DK' 
    if name == 'OffshoreWindPower': label = 'Offshore Wind Power in DK' 
    if name == 'GrossCon': 
        label = 'Gross Consumption in DK' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_DK1': 
        label = 'Gross Consumption in DK1' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_DK2': 
        label = 'Gross Consumption in DK2' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_BO': 
        label = 'Gross Consumption in Bornholm' 
    if name == 'alpha_TotalWindPower': label = r'$\alpha_{wind}$ in DK' 
    if name == 'alpha_TotalWindPower_DK1': label = r'$\alpha_{wind}$ in DK1' 
    if name == 'alpha_TotalWindPower_DK2': label = r'$\alpha_{wind}$ in DK2' 
    if name == 'alpha_TotalRenPower': label = r'$\alpha_{\rm VRE}$ in DK' 
    if name == 'alpha_TotalRenPower_DK1': label = r'$\alpha_{\rm VRE}$ in DK1' 
    if name == 'alpha_TotalRenPower_DK2': label = r'$\alpha_{\rm VRE}$ in DK2' 
    if name == 'alpha_TotalRenPower_BO': label = r'$\alpha_{\rm VRE}$ in BO' 
    if name == 'SolarPower_BO': label = 'Solar Power in Bornholm' 
    if name == 'BioPower_BO': label = 'Bio Power in Bornholm' 
    if name == 'Import_BO': label = 'Import/export in Bornholm' 


    return(label)

def handle_args(kwargs,argkeys_needed,verbose=False):
    """ Make custom namespace for specific method
    """

    args                =   Namespace(add=False,\
                            alpha=1,\
                            alpha_wind_cuts=[10,90],\
                            alpha_cuts=False,\
                            
                            bal_price_name='',\
                            bins=200,\
                            
                            col_indices='',\
                            col_names=[],\
                            color='b',\
                            colors=['b','r','k'],\
                            compare=False,\
                            comp_dev=50,\
                            cons_ob='',\

                            d_data='../../data/',\
                            data_df=False,\
                            data_type='xml',\
                            depth = False,\
                            depth_range = [-50,400],\
                            df_cols='',\
                            duration_ramp_min = 60,\
                            duration_range = [0,8*60*60],\
                            duration_cuts = False,\

                            exc_dev=1000,\
                            epoch=-1,\

                            fig_name=False,\
                            file_path='../../data',\
                            fig_format='pdf',\
                            freq_cut=[False],\
                            freq_cuts=[False],\
                            
                            histogram_values = [],\

                            include_storms = False,\
                            int_power = [],\

                            label='',\
                            labels=False,\
                            legend=False,\
                            load_names=[],\
                            log=False,\
                            ls=['-'],\

                            magnitude_range=[-2000,2000],\
                            max_val=1000,\
                            max_ramp=30,\
                            max_step=False,\
                            method='append',\
                            min_dip=-30,\

                            new_NaN_value=0,\
                            new_y_axis=False,\
                            number_of_storms=100,\

                            offset = 10,\
                            offsets = [1,2,3,4],\
                            percent = 95,
                            power_name='TotalWindPower',\
                            power_names=[],\
                            price_name='',\

                            raw_data_names='',\
                            remove_zeros=False,\

                            scale_by_capacity=False,\
                            SDAtype='duration',\
                            secs_or_minutes='secs',\
                            skiprows=1,\
                            spot_price_name='',\

                            test=False,\
                            test_plot=False,\
                            time='',\
                            time_cut=False,\
                            time_diff='year',\
                            time_period='year',\
                            times=False,\
                            two_axes=False,\

                            verbose=True,\
                            
                            xlab='',\
                            xlim=False,\
                            xlog=False,\

                            yearly=False,\
                            years=[2018],\
                            ylim=False,\
                            ylog=False,\
                            ylab='Power [MW]',\

                            zlog=False,\

                            )

    # Fill up new empty dictionary
    for key in argkeys_needed:
        if key in kwargs: 
            setattr(args,key,kwargs[key])
        else:
            # args[key] = getattr(default_args,key)
            if verbose: print('"%s" argument not passed, using default value' % key)

    return(args)
