###
### Submodule power
###

print('windpype submodule "power" imported')

import windpype.aux as aux
import windpype.capacity as cap
import numpy as np
import pandas as pd
import dateutil.parser
import xml.etree.ElementTree as ET 
from scipy import signal
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import colors
import seaborn as sns
import datetime as dt
from iteration_utilities import duplicates
import sys as sys


d_plot = 'plots/'

class PowerData():
    ''' This class defines an object that contains the time series of power produced and consumed from a specific dataset (e.g. DK1 or Bornholm). 
    '''

    def __init__(self,**kwargs):

        # handle default values and kwargs
        args                =   dict(file_path='',file_name='',ext='')
        args                =   aux.update_dictionary(args,kwargs)

        self.file_path      =   args['file_path']
        self.file_name      =   args['file_name']
        self.ext            =   args['ext']

    def info(self,verbose=True):
        ''' Prints basic info about this dataset.
        '''

        data_df = self.data_df
        if verbose:
            print('\n--------')
            print('Data object contains:')
            print('%s data points' % len(data_df))
            print('from %s to %s' % (np.min(data_df.datetime),np.max(data_df.datetime)))
            print('Minimum time step: %s sec' % (np.min(data_df.time_steps.values)))
            print('Maximum time step: %s sec' % (np.max(data_df.time_steps.values)))
            print('Most common time step: %s sec' % (np.median(data_df.time_steps.values)))
            print('--------')

        return(np.min(data_df.datetime),np.max(data_df.datetime))

    ### Data analysis

    def AddYearlyData(self):
        ''' Aggregates data (anything not time related) for one year at a time,
        and adds result as attribute.
        '''

        data_df = self.data_df.copy()
        datetimes = data_df['datetime']
        delta_time = data_df['delta_time'].values/60/60

        years = np.unique(np.array([_.year for _ in datetimes]))
        time_periods = [0]*len(years)
        for i,year in enumerate(years):
            time_periods[i] = [dt.datetime(year, 1, 1),dt.datetime(year+1, 1, 1)]

        col_names = list(data_df.keys())
        [col_names.remove(_) for _ in ['datetime','delta_time','time_steps']]

        yearly_data = pd.DataFrame({'years':years})
        for col_name in col_names:
            data = data_df[col_name].values
            yearly_data_temp = np.zeros(len(time_periods))
            for i,time_period in enumerate(time_periods):
                mask = np.array([(datetimes >= np.datetime64(time_period[0])) & (datetimes <= np.datetime64(time_period[1]))])[0]
                data_cut = data[mask]
                delta_time_cut = delta_time[mask]
                yearly_data_temp[i] = np.trapz(data_cut[data_cut > 0],delta_time_cut[data_cut > 0])
            yearly_data[col_name] = yearly_data_temp

        self.yearly_data = yearly_data

    def AddHourlySteps(self,**kwargs):
        ''' A raw method that only extracts changes every 1 hour, must be used with hourly data.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','include_storms','scale_by_capacity','epoch','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        # if a.include_storms: print('Now extracting hourly steps for %s, with storms' % a.col_name)
        # if not a.include_storms: print('Now extracting hourly steps for %s, without storms' % a.col_name)
        if a.verbose: print('Calculating hourly steps for %s' % a.col_name)

        data_df = self.data_df.copy()
        if a.scale_by_capacity:
            if 'accum_cap_'+aux.zone(a.col_name) not in data_df.keys(): 
                print('Calculating accumulated capacity for %s' % a.col_name)
                self.AddAccumCapacity(col_name=a.col_name)
            data_df = self.data_df.copy()

        if a.epoch != -1:
            print('Picking out an epoch')


        # Check that data is hourly based:
        if max(data_df['time_steps']) != min(data_df['time_steps']):
            sys.exit('time steps are not constant, cannot proceed')
        if max(data_df['time_steps']) != 3600:
            sys.exit('time step is not 1 hour, cannot proceed')

        if max(data_df[a.col_name].values) > -1: 
            data = data_df[a.col_name].values
            raw_steps = np.array([data[_+1] - data[_] for _ in range(len(data)-1)])
            raw_steps = np.append(raw_steps,0)
            datetimes = data_df['datetime'].values
            # Take out storms?
            if not a.include_storms:
                datetime_storms = aux.identify_storms()
                for datetime_storm in datetime_storms:
                    delta_time = np.abs((datetimes - datetime_storm)/np.timedelta64(1,'D'))
                    if min(delta_time) <= 1:
                        raw_steps[delta_time <= 1] = 0
            # Remove the first ramp up from 0
            if raw_steps[0] == 0:
                indices = np.arange(len(raw_steps))
                raw_steps[indices[np.where(raw_steps != 0)][0]] = 0
            data_df['hourly_raw_steps_'+a.col_name] = raw_steps
            # Normalize hourly steps
            if a.scale_by_capacity:
                # Calculate ramps relative to the installed capacity at the time
                capacity_cut = data_df['accum_cap_'+aux.zone(a.col_name)].values
                steps = raw_steps/capacity_cut*100.
                data_df['hourly_steps_'+a.col_name] = steps
            if 'Price' in a.col_name: 
                # Calculate ramps relative to spot price at the time
                if 'DK1' in a.col_name: spot_price = data_df['SpotPriceEUR_DK1'].values
                if 'DK2' in a.col_name: spot_price = data_df['SpotPriceEUR_DK2'].values
                # if 'DK1' in a.col_name: spot_price = data_df['SpotPriceEUR'].values
                # indices_to_replace = list(np.where(spot_price == 0)[0])
                # spot_price[indices_to_replace] = spot_price[[_-1 for _ in indices_to_replace]]
                # print(min(data_df['SpotPriceEUR'].values),np.max(data_df['SpotPriceEUR'].values),min(raw_steps),np.max(raw_steps))
                steps = raw_steps/spot_price*100. 
                steps[spot_price == 0] = 0
                data_df['hourly_steps_'+a.col_name] = steps

            if 'Con' in a.col_name: 
                # Calculate ramps relative to consumption at the time
                cons = data_df[a.col_name].values
                steps = raw_steps/cons*100.
                steps[cons == 0] = 0
                data_df['hourly_steps_'+a.col_name] = steps
        else:
            print('No data in this time period, could not calculate hourly steps...')

        self.data_df = data_df

    def AddPSD(self,**kwargs):
        ''' Calculates Power Spectral Distribution (PSD) of wind power, 
        and adds result as attribute.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        PSDomega,PSD        =   signal.periodogram(self.data[a.power_name], 1)

        setattr(self,a.power_name+'_PSDomega',PSDomega)
        setattr(self,a.power_name+'_PSD',PSD)

    def AddFFT(self,**kwargs):
        ''' Calculates Fast Fourier Transform (FFT) for wind power and inserts it back into dataframe data_df. Returns dataframe.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','verbose','test']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        power               =   data_df[a.power_name].values
        mean_power          =   np.mean(power[power != -1])
        FFT                 =   np.fft.fft(power-mean_power)
        dT                  =   data_df.delta_time[1]-data_df.delta_time[0] #s
        if a.verbose: print('Time step used for FFT: %s sec' % dT)
        freq                =   np.fft.fftfreq(data_df.delta_time.shape[-1])*1/dT
        # omega               =   freq/(2*np.pi)

        data_df[a.power_name+'_freq'] = freq
        data_df[a.power_name+'_FFT'] = FFT

        self.data_df        =   data_df

    def AddiFFT(self,**kwargs):
        ''' Calculates inverse Fast Fourier Transform (iFFT) for dataframe data_df 
        and inserts it back into dataframe data_df. 
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','data_df','freq_cut','verbose','test']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.verbose: print('Calculating iFFT')

        self.AddFFT(power_name=a.power_name,verbose=a.verbose)
        data_df             =   self.data_df.copy()

        if a.test:
            a.col_name = a.test
            data_df[a.test + '_freq'] = data_df[a.power_name+'_freq']
            if a.test == 'Kolmogorov':
                spectrum = np.abs(data_df[a.test + '_freq'])**(-5./3)
                data_df[a.test + '_FFT'] = spectrum * np.sqrt(abs(data_df[a.test + '_freq']))
                data_df[a.test + '_FFT'][data_df[a.test + '_freq'] == 0] = np.max(data_df[a.test + '_FFT'][data_df[a.test + '_freq'] != 0])

        # Then get inverse FFT
        freq                =   data_df[a.power_name+'_freq'].values
        FFT                 =   data_df[a.power_name+'_FFT'].values
        if a.test:
            FFT                 =   data_df[a.test+'_FFT'].values

        if len(a.freq_cut) > 1: 
            a.freq_cut          =   np.sort(a.freq_cut)
            if a.verbose: print('For frequency cut: %.2e to %.2e Hz' % (a.freq_cut[0],a.freq_cut[1]))
            # omega_cut           =   a.freq_cut/(2*np.pi)
            FFT_cut             =   np.copy(FFT)
            # print(min(abs(freq)),max(abs(freq)))
            FFT_cut[abs(freq) < a.freq_cut[0]] = 0
            FFT_cut[abs(freq) > a.freq_cut[1]] = 0
            # test = np.abs(freq[FFT_cut != 0])
            # print(min(test),max(test))

            # if omega_cut[0] < 0: 
                # print('Low (neg. durations), high freq')
                # FFT_cut[abs(omega) > omega_cut[1]] = 0
                # FFT_cut[omega < 0] = 0
            # if omega_cut[0] > 0: 
                # FFT_cut[abs(omega) < omega_cut[1]] = 0
                # print('long durations, short freq')
            # if omega_cut[0] > 0: 
                # FFT_cut[abs(omega) < omega_cut[0]] = 0
                # FFT_cut[abs(omega) > omega_cut[1]] = 0
            # print(len(FFT_cut[(abs(omega) < omega_cut[0]) | (abs(omega) > omega_cut[1])]))
            if a.verbose: print('Frequency cut contains %.2f %% of FFT power (real part)' % (np.sum(np.abs(FFT_cut.real))/np.sum(np.abs(FFT.real))*100.))
            iFFT            = np.fft.ifft(FFT_cut)
        else:
            iFFT            = np.fft.ifft(FFT)

        if a.test:
            data_df[a.test+'_iFFT'] = iFFT
        else:
            # Setting signal to 0 where there is no data
            iFFT[data_df[a.power_name] == -1] = 0
            data_df[a.power_name+'_iFFT'] = iFFT
        
        self.data_df            =   data_df
        if a.verbose: print('------')

    def AddExceptionTest(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','exc_dev','test_plot','time_cut']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df             =   self.data_df.copy()

        if a.test_plot:
            # TEST, cutting out part of data
            data_df = data_df[np.array([(data_df['datetime'].values > a.time_cut[0]) & (data_df['datetime'].values < a.time_cut[1])])[0]]

        datetime            =   data_df['datetime'].values
        power               =   data_df[a.power_name].values

        # Calculate deviations from previous data point:
        new_segment         =   True
        index               =   0
        exc_keep            =   np.array(len(data_df)*[False])
        for _ in range(2*len(data_df)):
            # print(index)
            if new_segment:
                # print('new segment!!')
                # Just calculate min and max values
                min_val,max_val =   power[index]-a.exc_dev,power[index]+a.exc_dev
                new_segment     =   False
            else:
                # Calculate if point is outside valid range:
                inside          =   (min_val < power[index] < max_val)
                # print(min_val,power[index],max_val)
                if not inside:
                    # print('not inside')
                    exc_keep[index] = True
                    exc_keep[index-1] = True
                    new_segment = True
                    index -= 1
                if inside:
                    pass
                    # print('inside')

            index += 1
            if index > len(data_df)-1: break

        exc_keep[0] = True
        exc_drop = [exc_keep == False]

        self.exc_keep       =   exc_keep
        self.exc_drop       =   exc_drop

        if a.test_plot:
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(2,1,1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Power [MW]')
            self.PlotTimeSeries(col_names=[a.power_name],colors=['k'],labels=[a.power_name],\
                legend=False,ylab='Power [MW]',add=True,time_cut=a.time_cut)
            ax1 = plt.gca()
            ax1.plot(datetime[self.exc_drop],power[self.exc_drop],'ro',label='dropped after exception test')
            ax1.plot(datetime[self.exc_keep],power[self.exc_keep],'bo',label='kept after exception test')
            plt.legend()

    def AddSDA(self,**kwargs):
        ''' Runs a Swinging Door Algorithm on the power data to look for ramps up and down.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','comp_dev','test_plot','time_cut','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        print('\nNow applying Swinging Door Algorithm to %s data' % a.power_name)
        
        data_df = self.data_df.copy()

        if a.test_plot:
            # TEST, cutting out part of data
            data_df = data_df[np.array([(data_df['datetime'].values > a.time_cut[0]) & (data_df['datetime'].values < a.time_cut[1])])[0]]

        datetime = data_df['datetime'].values
        delta_time = data_df['delta_time'].values
        power = data_df[a.power_name].values

        # Remove unnecessary data
        self.AddExceptionTest(**kwargs)

        print('original dataframe of %s rows' % len(data_df))
        red_data_df         =     data_df[self.exc_keep]
        print('reduced to %s rows after exception test' % len(red_data_df))
        delta_time = red_data_df['delta_time'].values
        power = red_data_df[a.power_name].values
        datetime = red_data_df['datetime'].values

        indices             =   []
        index               =   0
        recent_archive_time =   delta_time[0]
        for _ in range(2*len(red_data_df)):
            # print('\n',index)
            incoming_value      =   power[index]
            time                =   delta_time[index]
            time_step           =   time - recent_archive_time

            if index == 0:
                recent_archive_value = incoming_value
                recent_archive_time = time
                indices.append(index)
                # print('Very first point, keeping this index %s' % index)
                new_segment         =   True
            else:
                if new_segment:
                    if time_step == 0:
                        # print('Very first point in this new segment, only storing value')
                        recent_archive_value = incoming_value
                        recent_archive_time = time
                    else:
                        # print('First time step in new segment! Not sure yet, if keeping this index (%s)' % index)
                        current_snapshot_value = incoming_value #snapshot for next time...
                        max_slope       =   (incoming_value+a.comp_dev - recent_archive_value)/time_step
                        min_slope       =   (incoming_value-a.comp_dev - recent_archive_value)/time_step
                        # print('Slopes:')
                        # print(min_slope,max_slope)
                        new_segment = False
                else:
                    # Calculate ref slope
                    ref_slope       =   (incoming_value - recent_archive_value)/time_step
                    # print('ref slope after %s sec: %s' % (time_step,ref_slope))
                    # print(min_slope,max_slope,ref_slope)
                    if not (min_slope < ref_slope < max_slope):
                        # print('point at index %s outside swinging door, go back one to start a new segment' % index)
                        # Start new segment with this incoming value as new archive value
                        recent_archive_value = power[index-1]
                        recent_archive_time = delta_time[index-1]
                        # time_step           =   time - recent_archive_time
                        new_segment         =   True
                        indices.append(index-1)
                        index               -=  2
                    else:
                        # print('point at index %s within swinging door, set as new snapshot value')
                        # print('Previous index %s NOT kept' % (index-1))
                        current_snapshot_value = incoming_value #snapshot for next time...
                        # Do not store this index, but update max and min slopes
                        new_max_slope       =   (incoming_value+a.comp_dev - recent_archive_value)/time_step
                        new_min_slope       =   (incoming_value-a.comp_dev - recent_archive_value)/time_step
                        # Making sure that the opening narrows
                        max_slope           =   min(max_slope,new_max_slope)
                        min_slope           =   max(min_slope,new_min_slope)
                        # print('new slopes:')
                        # print(min_slope,max_slope)


                    # a = asf 
            index           +=  1
            if index > len(red_data_df)-1: break

        print('Identified about %s ramps ' % (len(indices)))

        times_ADS       =   datetime[indices]
        power_ADS       =   power[indices]

        ramps           =   np.array([power[indices[_+1]] - power[indices[_]] for _ in range(len(indices)-1)])
        ramp_index      =   np.arange(len(times_ADS)-1)
        ramp_up_index   =   ramp_index[ramps > 0]
        ramp_do_index   =   ramp_index[ramps < 0]

        times_up_ramps  =   [[times_ADS[_],times_ADS[_+1]] for _ in ramp_up_index]
        power_up_ramps  =   [[power_ADS[_],power_ADS[_+1]] for _ in ramp_up_index]
        times_do_ramps  =   [[times_ADS[_],times_ADS[_+1]] for _ in ramp_do_index]
        power_do_ramps  =   [[power_ADS[_],power_ADS[_+1]] for _ in ramp_do_index]

        if a.test_plot:
            fig = plt.gcf()
            ax1 = fig.add_subplot(2,1,2)
            ax1.plot(times_ADS,power_ADS,'x',ms=15,mew=2,color='purple',label='SDA selected')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Power [MW]')
            for _ in range(len(times_up_ramps)):
                ax1.plot(times_up_ramps[_],power_up_ramps[_],'--',color='g')
            for _ in range(len(times_do_ramps)):
                ax1.plot(times_do_ramps[_],power_do_ramps[_],'--',color='r')
            plt.legend()
            if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

        durations_min = np.array([[times_ADS[_+1]-times_ADS[_]] for _ in ramp_index])/np.timedelta64(1, 'm')
        setattr(self,a.power_name+'_SDA_ramp_durations_min',durations_min.flatten())
        setattr(self,a.power_name+'_SDA_ramp_magnitudes',ramps)
        setattr(self,a.power_name+'_SDA_times_up_ramps',times_up_ramps)
        setattr(self,a.power_name+'_SDA_power_up_ramps',power_up_ramps)
        setattr(self,a.power_name+'_SDA_times_ramps',times_ADS[0:-1])

    def AddAccumCapacity(self,**kwargs):
        '''

        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','d_data']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df = self.data_df.copy()
        min_time,max_time = np.min(data_df.datetime),np.max(data_df.datetime)

        # Load capacities
        capacity_ob = cap.Capacity(file_path=a.d_data+'ENS/',file_name='capacity_df')
        capacity_ob.AddData(file_path=a.d_data+'ENS/')

        capacity_df_or = capacity_ob.cap_data_df # capacity,date = capacity_ob.data_df_DK1['capacity'],capacity_ob.data_df_DK1['date']

        if 'DK1' in a.col_name: cap_name = 'accum_cap_DK1'
        if 'DK2' in a.col_name: cap_name = 'accum_cap_DK2'
        if 'BO' in a.col_name: cap_name = 'accum_cap_BO'
        if a.col_name == 'TotalWindPower': cap_name = 'accum_cap_DK'
        if a.col_name == 'OnshoreWindPower': cap_name = 'accum_cap_onshore'
        if a.col_name == 'OffshoreWindPower': cap_name = 'accum_cap_offshore'

        capacity_df = capacity_df_or[['datetime',cap_name]].copy()

        # Rename capacity columns to contain column name
        for key in capacity_df.keys():
            if key != 'datetime':
                new_key = key+'_'+a.col_name
                capacity_df[new_key] = capacity_df_or[key]
                # capacity_df.drop(key,axis=1)

        # Merge capacities into data
        capacity_df = capacity_df.drop_duplicates(subset='datetime',keep='last')
        # capacity_df['datetime'] = capacity_df['datetime'].values + np.timedelta64(0,'s')
        data_df_w_capacity = pd.merge(data_df,capacity_df,on='datetime',how='outer')
        mask = np.array([(data_df_w_capacity.datetime >= min_time) & (data_df_w_capacity.datetime <= max_time)])[0]
        data_df_w_capacity = data_df_w_capacity[mask]
        # update times and remove any duplicates
        data_df_w_capacity = aux.update_times(data_df_w_capacity)
        data_df_w_capacity = data_df_w_capacity.drop_duplicates(subset='datetime')

        # fill missing values
        data_df_w_capacity[cap_name] = data_df_w_capacity[cap_name].ffill() # Continue with last measured total capacity
        data_df_w_capacity[cap_name] = data_df_w_capacity[cap_name].ffill()
        data_df_w_capacity[cap_name] = data_df_w_capacity[cap_name].bfill()

        data_df = data_df_w_capacity
        data_df = aux.update_times(data_df)

        self.data_df = data_df

    def AddPenetrationFraction(self,**kwargs):
        ''' Calculates wind penetration in percent of total consumption, and adds result in dataframe.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_names','load_names','yearly']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        datetimes = data_df['datetime']
        delta_time = data_df['delta_time'].values/60/60

        if a.yearly:
            print('Calculating penetration fraction for every year')
            years = np.unique(np.array([_.year for _ in datetimes]))
            time_periods = [0]*len(years)
            for i,year in enumerate(years):
                time_periods[i] = [dt.datetime(year, 1, 1),dt.datetime(year+1, 1, 1)]
            alpha = pd.DataFrame({'years':years})
            for power_name,load_name in zip(a.power_names,a.load_names):
                power = data_df[power_name]
                load = data_df[load_name]
                alpha_temp = np.zeros(len(time_periods))
                for i,time_period in enumerate(time_periods):
                    mask = np.array([(datetimes >= np.datetime64(time_period[0])) & (datetimes <= np.datetime64(time_period[1]))])[0]
                    power_cut = power[mask]
                    load_cut = load[mask]
                    delta_time_cut = delta_time[mask]
                    if len(load_cut[load_cut > 0]) > 3000:
                        if np.sum(load_cut[power_cut > 0]) == 0: 
                            alpha_temp[i] = 0
                        else: 
                            power_int = np.trapz(power_cut[power_cut > 0],delta_time_cut[power_cut > 0])
                            load_int = np.trapz(load_cut[power_cut > 0],delta_time_cut[power_cut > 0])
                            alpha_temp[i] = np.sum(power_int)/np.sum(load_int)*100. # %
                    else:
                        print('Not enough load data for this year:')
                        print(time_period)
                alpha[power_name] = alpha_temp

            self.alpha = alpha
        else:
            print('Calculating penetration fraction for every time step')
            for power_name,load_name in zip(a.power_names,a.load_names):
                alpha_temp = data_df[power_name].values/data_df[load_name].values*100. 
                alpha_temp[data_df[load_name].values == -1] = -1
                alpha_temp[data_df[load_name].values == 0] = -1
                data_df['alpha_' + power_name] =  alpha_temp
            self.data_df = data_df
        print('------')

    def AddHighPenetrationIndex(self,**kwargs):
        ''' Add index to separate high wind penetration from low
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['alpha_cuts','power_name','load_names','yearly']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        # calculate wind penetration fraction
        if 'alpha_' + a.power_name not in data_df.keys():
            if 'DK1' in a.power_name: load_name = 'GrossCon_DK1'
            if 'DK2' in a.power_name: load_name = 'GrossCon_DK2'
            if 'BO' in a.power_name: load_name = 'GrossCon_BO'
            if a.power_name == 'TotalWindPower': load_name = 'GrossCon'
            if a.power_name == 'TotalRenPower': load_name = 'GrossCon'
            print('Using load: '+load_name)
            self.AddPenetrationFraction(power_names=[a.power_name],load_names=[load_name])
            data_df = self.data_df.copy()

        # High wind penetration
        alpha = data_df['alpha_' + a.power_name].values
        high_penetration_index = np.zeros(len(data_df)) + 1
        high_penetration_index[alpha < a.alpha_cuts[1]] = 0
        high_penetration_index[alpha == -1] = -1

        data_df['high_penetration_index_' + a.power_name] = high_penetration_index
        self.data_df = data_df

        percent = len(high_penetration_index[high_penetration_index == 1])/len(high_penetration_index[high_penetration_index >= 0])*100.
        print('Penetration fraction is above %s%% %.2f%% of the time' % (a.alpha_cuts[1],percent))
        print('With %s data points' % (len(high_penetration_index[high_penetration_index == 1])))
        print('Out of %s data points' % (len(high_penetration_index)))

        change_in_index = np.array([high_penetration_index[_+1]-high_penetration_index[_] for _ in range(len(data_df)-1)])
        change_in_index = np.append(change_in_index,0)
        change_in_index[high_penetration_index == -1] = 0
        while data_df.datetime.values[change_in_index == -1][0] < data_df.datetime.values[change_in_index == 1][0]:
            change_in_index[change_in_index == -1] = np.append(0,change_in_index[change_in_index == -1][1::])
            # print(data_df.datetime.values[change_in_index == -1][0],data_df.datetime.values[change_in_index == 1][0])
        # durations = data_df.datetime.values[change_in_index == -1] - data_df.datetime.values[change_in_index == 1][:-1]
        end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1]

        while len(start_times) != len(end_times):
            if len(start_times) > len(end_times):
                end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1][:-1]
            if len(end_times) > len(start_times):
                end_times,start_times = data_df.datetime.values[change_in_index == -1][:-1], data_df.datetime.values[change_in_index == 1]
        try:
            durations = end_times - start_times
        except ValueError:
            print('could not calculate durations')
        durations = durations/np.timedelta64(1,'h')
        print('Min and maximum epoch durations of high penetration: %f and %f hrs' % (min(durations),max(durations)))
        setattr(self,'high_penetration_duration_hrs',durations)
        setattr(self,'high_penetration_start_times',start_times)
        setattr(self,'high_penetration_end_times',end_times)

        # Low wind penetration
        alpha = data_df['alpha_' + a.power_name].values
        low_penetration_index = np.zeros(len(data_df)) + 1
        low_penetration_index[alpha > a.alpha_cuts[0]] = 0
        low_penetration_index[alpha == -1] = -1

        data_df['low_penetration_index_' + a.power_name] = low_penetration_index
        self.data_df = data_df

        percent = len(low_penetration_index[low_penetration_index == 1])/len(low_penetration_index[low_penetration_index >= 0])*100.
        print('Penetration fraction is below %s%% %.2f%% of the time' % (a.alpha_cuts[0],percent))
        print('With %s data points' % (len(low_penetration_index[low_penetration_index == 1])))
        print('Out of %s data points' % (len(low_penetration_index)))

        change_in_index = np.array([low_penetration_index[_+1]-low_penetration_index[_] for _ in range(len(data_df)-1)])
        change_in_index = np.append(change_in_index,0)
        change_in_index[low_penetration_index == -1] = 0
        while data_df.datetime.values[change_in_index == -1][0] < data_df.datetime.values[change_in_index == 1][0]:
            change_in_index[change_in_index == -1] = np.append(0,change_in_index[change_in_index == -1][1::])
            # print(data_df.datetime.values[change_in_index == -1][0],data_df.datetime.values[change_in_index == 1][0])
        # durations = data_df.datetime.values[change_in_index == -1] - data_df.datetime.values[change_in_index == 1][:-1]
        end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1]
        if len(start_times) > len(end_times):
            end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1][:-1]
        if len(end_times) > len(start_times):
            end_times,start_times = data_df.datetime.values[change_in_index == -1][:-1], data_df.datetime.values[change_in_index == 1]
        try:
            durations = end_times - start_times
        except ValueError:
            print('could not calculate durations')
        durations = durations/np.timedelta64(1,'h')
        print('Min and maximum epoch durations of low penetration: %f and %f hrs' % (min(durations),max(durations)))
        setattr(self,'low_penetration_duration_hrs',durations)
        setattr(self,'low_penetration_start_times',start_times)
        setattr(self,'low_penetration_end_times',end_times)
        print('------')

    def IdentifyStorms(self,**kwargs):
        ''' Identify storms, based on extreme hourly ramps
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','number_of_storms','include_storms']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        if not hasattr(self,'hourly_raw_steps_'+a.col_name): 
            self.AddHourlySteps(**kwargs)
        hourly_steps = data_df['hourly_steps_'+col_name]
        datetimes = data_df['datetime']

        # Sort according to step size
        datetimes = datetimes[hourly_raw_steps.argsort()[::-1]]
        hourly_raw_steps = hourly_raw_steps[hourly_raw_steps.argsort()[::-1]]

        # Select the most powerful
        datetimes = datetimes[0:a.number_of_storms]
        hourly_raw_steps = hourly_raw_steps[0:a.number_of_storms]


        # Sort according to date
        hourly_raw_steps = hourly_raw_steps[datetimes.argsort()[::-1]]
        datetimes = datetimes[datetimes.argsort()[::-1]]

        datetimes_with_storms = aux.search_for_storms(datetimes)

    ### Visualization

    def CompEpochs(self,**kwargs):
        ''' Compare data during and outside of high wind penetration epochs
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','colors','ls','fig_name','label','xlim','xlab','alpha','bins','add','compare','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df             =   self.data_df.copy()

        add = False
        _ = 0
        ext = [r'$\alpha_{\rm VRE}<20\%$',r'$\alpha_{\rm VRE}>80\%$']
        for col_name in a.col_names:
            low_penetration_index = data_df['low_penetration_index_'+col_name].values
            high_penetration_index = data_df['high_penetration_index_'+col_name].values
            label = aux.pretty_label(col_name)
            for epoch in [0,1]:
                print('Now doing epoch %s' % epoch)
                if a.compare:
                    if epoch == 0: data = data_df[a.compare+'_'+col_name][low_penetration_index == 1]
                    if epoch == 1: data = data_df[a.compare+'_'+col_name][high_penetration_index == 1]
                else:
                    if epoch == 0: data = data_df[col_name][low_penetration_index == 1]
                    if epoch == 1: data = data_df[col_name][high_penetration_index == 1]          
                self.PlotHisto(histogram_values=[0],data=data,colors=[a.colors[_]],ls=[a.ls[_]],alpha=a.alpha,\
                    add=add,labels=[ext[epoch]],ylog=True,xlim=a.xlim,xlab=a.xlab,bins=a.bins)
                _ += 1
                if _ == 1:
                    ax1 = plt.gca()
                    ylim = ax1.get_ylim()
                    ax1.plot([0,0],[ylim[0]-1,ylim[1]+1],'k--',lw=2)
                    ax1.set_ylim(ylim)
                # ax1.set_yscale('log')
                if _ >= 1: add=True

        ax1 = plt.gca()

        ax1.legend()
        # if a.label != '': ax1.legend()

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=300)

    def PlotMinimumRamps(self,col_names=[],colors=[],offsets=[],save=False,fig_name='min_ramps',d_plot=d_plot,time_cut=False,max_ramp=30):

        data_df = self.data_df.copy()
        time_step = np.median(data_df.time_steps)
        if time_step >= 60*60: time_step_str = '%s hour' % (time_step/60/60)
        if time_step < 60*60: time_step_str = '%s minute' % (time_step/60)

        print('Going to plot ramps of length %s' % time_step_str)

        fig = plt.figure(figsize=(18,10))
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.view_init(elev=30, azim=60)
        for i,col_name in enumerate(col_names):
            hourly_ramps,datetimes = self.GetMinimumRamps(col_name=col_name)
            if time_cut:
                mask = np.array([(datetimes > time_cut[0]) & (datetimes < time_cut[1])])[0]
                hourly_ramps,datetimes = hourly_ramps[mask],datetimes[mask]
            datetimes = datetimes[hourly_ramps != 0]
            hourly_ramps = hourly_ramps[hourly_ramps != 0]
            datetimes = datetimes[abs(hourly_ramps) < max_ramp]
            hourly_ramps = hourly_ramps[abs(hourly_ramps) < max_ramp]
            print('Max ramp for %s: %.5s' % (col_name,np.max(np.abs(hourly_ramps))))
            print('happened around: %s ' % datetimes[np.argmax(abs(hourly_ramps))])
            hist, bins = np.histogram(hourly_ramps, bins=200)
            xs = (bins[:-1] + bins[1:])/2
            width = xs[1]-xs[0]
            # Scale histogram down to percentage of area under the graph
            hist = hist*100/np.trapz(hist,x=xs)
            ax1.bar(xs,hist,zs=offsets[i],width=width,lw=0,fc=colors[i],zdir='y',alpha=0.5)
        ax1.set_xlim([-max_ramp,max_ramp])
        ax1.set_yticks(offsets)
        ax1.set_yticklabels(col_names)
        ax1.set_zlabel('Relative frequency')
        ax1.set_xlabel('\n%s ramps [%% of installed capacity]' % time_step_str)

        if save: plt.savefig(d_plot+fig_name,dpi=300)

    def PlotSDARamps(self,**kwargs):
        ''' 3D view of SDA ramps.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','colors','offsets','fig_name','time_cut','max_ramp','duration_ramp_min']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)
        print('\nGoing to plot ramps of length %s min' % a.duration_ramp)

        data_df = self.data_df.copy()

        fig = plt.figure(figsize=(18,10))
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.view_init(elev=30, azim=60)

        for i,col_name in enumerate(a.col_names):
            magnitudes = getattr(self,col_name+'_SDA_ramp_magnitudes')
            durations = getattr(self,col_name+'_SDA_ramp_durations_min')
            datetimes = getattr(self,col_name+'_SDA_times_ramps')
            print('Now plotting ramps for %s ' % col_name)
            magnitudes = magnitudes[durations == a.duration_ramp_min]
            datetimes = datetimes[durations == a.duration_ramp_min]
            durations = durations[durations == a.duration_ramp_min]

            if not hasattr(self,'ramps_'+col_name): 
                self.MagnitudesToRamps(magnitudes,datetimes,col_name=col_name,**kwargs)
            SDA_ramps = getattr(self,'ramps_'+col_name)

            print('Max ramp for %s: %.5s %%' % (col_name,np.max(np.abs(SDA_ramps))))
            print('happened around: %s ' % datetimes[np.argmax(abs(SDA_ramps))])
            hist, bins = np.histogram(SDA_ramps, bins=200)
            xs = (bins[:-1] + bins[1:])/2
            width = xs[1]-xs[0]
            # Scale histogram down to percentage of area under the graph
            hist = hist*100/np.trapz(hist,x=xs)
            ax1.bar(xs,hist,zs=a.offsets[i],width=width,lw=0,fc=a.colors[i],zdir='y',alpha=0.5)
        ax1.set_xlim([-a.max_ramp,a.max_ramp])
        ax1.set_yticks(a.offsets)
        ax1.set_yticklabels([aux.pretty_label(col_name) for col_name in a.col_names])
        ax1.set_zlabel('Relative frequency')
        ax1.set_xlabel('\n%s min ramps [%% of installed capacity]' % a.duration_ramp_min)

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotHourlySteps(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','labels','ylim','time_cut','add','ylab','legend','alpha','colors','fig_name','add']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        if a.add:
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(15,5))
            ax1 = fig.add_subplot(1,1,1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel(a.ylab)

        if a.labels == []:
            a.labels = [aux.pretty_label(col_name) for col_name in a.col_names]

        for col_name,color,label in zip(a.col_names,a.colors,a.labels):

            if not hasattr(self,'hourly_steps_'+col_name): 
                self.AddHourlySteps(col_name=col_name,**kwargs)
            # hourly_steps = getattr(self,'hourly_raw_steps_'+col_name)
            hourly_steps = data_df['hourly_steps_'+col_name]
            datetimes = data_df['datetime']
            if a.time_cut:
                mask = np.array([(datetimes > a.time_cut[0]) & (datetimes < a.time_cut[1])])[0]
                hourly_steps,datetimes = hourly_steps[mask],datetimes[mask]
            ax1.plot(datetimes,hourly_steps,color,label=label,alpha=a.alpha)

        if a.ylim: 
            ax1.set_ylim(a.ylim)

        if a.legend: fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes) # this creates a combined legend box for both axes

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotHourlyStepProfiles(self,**kwargs):
        ''' 3D view of hourly ramps.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','colors','offsets','fig_name','time_cut','max_ramp','zlog','include_storms','alpha']
        a                   =   aux.handle_args(kwargs,argkeys_needed)
        
        data_df = self.data_df.copy()

        if np.median(data_df.time_steps) != 60*60:
            sys.exit('time step is not 1 hour, cannot proceed')

        data_df = self.data_df.copy()

        fig = plt.figure(figsize=(18,10))
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.view_init(elev=30, azim=60)
        for i,col_name in enumerate(a.col_names):
            if not hasattr(self,'hourly_steps_'+col_name): 
                self.AddHourlySteps(col_name=col_name,scale_by_capacity=True,**kwargs)
            data_df = self.data_df.copy()
            hourly_steps = data_df['hourly_steps_'+col_name].values
            datetimes = data_df['datetime']
            if a.time_cut:
                mask = np.array([(datetimes > a.time_cut[0]) & (datetimes < a.time_cut[1])])[0]
                hourly_steps,datetimes = hourly_steps[mask],datetimes[mask]
            print('Max ramp for %s: %.5s' % (col_name,hourly_steps[np.argmax(np.abs(hourly_steps))]))
            print('happened around: %s ' % datetimes[np.argmax(abs(hourly_steps))])
            hourly_steps = hourly_steps[hourly_steps != 0]
            hourly_steps = hourly_steps[(hourly_steps > -a.max_ramp) & (hourly_steps < a.max_ramp)]
            hist, bins = np.histogram(hourly_steps, bins=200)
            xs = (bins[:-1] + bins[1:])/2
            width = xs[1]-xs[0]
            # Scale histogram down to percentage of area under the graph
            hist = hist*100/np.trapz(hist,x=xs)
            ax1.bar(xs,hist,zs=a.offsets[i],width=width,lw=0,fc=a.colors[i],zdir='y',alpha=a.alpha,zorder=0)
        ax1.set_xlim([-a.max_ramp,a.max_ramp])
        ax1.set_yticks(a.offsets)
        if a.zlog: ax1.set_zscale('log')
        ax1.set_yticklabels([aux.pretty_label(col_name) for col_name in a.col_names])
        ax1.set_zlabel('Relative frequency')
        ax1.set_xlabel('\nHourly ramps [% of installed capacity]')
        if 'Price' in col_name: ax1.set_xlabel('\nHourly ramps [%% of spot price]')

        # ax1.set_rasterized(True)
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='png', dpi=300, transparent=False) #rasterized=True, 

    def PlotHourlyStepSTD(self,**kwargs):
        ''' Plots Standard Deviation of histograms of changes every 1 hour, must be used with 1 hour time series data.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['percent','col_names','colors','offsets','fig_name','time_period','include_storms','alpha','labels']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        STD_df = self.GetHourlyStepSTD(**kwargs)

        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1,1,1)
        STD_df = STD_df.set_index('time')
        if a.time_period == 'seasons':
            STD_df[[a.col_names[0]+'_winter',a.col_names[0]+'_spring',a.col_names[0]+'_summer',a.col_names[0]+'_fall']].plot(ax=ax1,kind='bar',alpha=a.alpha,color=a.colors)
        else:
            STD_df[a.col_names].plot(ax=ax1,kind='bar',alpha=a.alpha,color=a.colors) 
        ax1.set_ylabel('$\sigma$ of hourly ramps [% of installed capacity]')

        if not a.labels:
            a.labels = [aux.pretty_label(col_name,percent=True) for col_name in a.col_names]
        handles, labels = ax1.get_legend_handles_labels()
        # plt.legend(handles[::-1],a.labels[::-1],fontsize=13,bbox_to_anchor=(1., 1.23))
        plt.legend(handles[::-1],a.labels[::-1],fontsize=13,loc='upper left',fancybox=True)

        if a.fig_name: 
            plt.savefig(d_plot+'%s' % a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotHourlyStepPerc(self,**kwargs):
        ''' Plots percentiles of histograms of changes every 1 hour, must be used with 1 hour time series data.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['percent','col_names','colors','offsets','fig_name','time_period','include_storms','alpha','labels']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        perc_dict = self.GetHourlyStepPerc(**kwargs)
        # fig = plt.figure(figsize=(15,6))
        fig = plt.figure(figsize=(10,6))
        panel = 1
        for name,percent in zip(['lower','upper'],[100-a.percent,a.percent]):
            ax1 = fig.add_subplot(1,2,panel)
            perc_df = perc_dict[name]
            perc_df = perc_df.set_index('time')
            if a.time_period == 'seasons':
                perc_df[[a.col_names[0]+'_winter',a.col_names[0]+'_spring',a.col_names[0]+'_summer',a.col_names[0]+'_fall']].plot(ax=ax1,kind='barh',alpha=a.alpha,color=a.colors)
            else:
                perc_df[a.col_names].plot(ax=ax1,kind='barh',alpha=a.alpha,color=a.colors)
            if panel == 1: 
                ax1.legend_.remove()
            if panel == 2: 
                ax1.yaxis.set_label_position("right")
                ax1.yaxis.tick_right()
            ax1.set_ylabel('%s percentile of hourly ramps [%% of installed capacity]' % percent)
            panel += 1
        plt.subplots_adjust(wspace=0.04,top=0.75)

        if not a.labels: a.labels = [aux.pretty_label(col_name,percent=True) for col_name in a.col_names]
        handles, labels = ax1.get_legend_handles_labels()
        # plt.legend(handles[::-1],a.labels[::-1],fontsize=13,bbox_to_anchor=(1., 1.23))
        plt.legend(handles[::-1],a.labels[::-1],fontsize=13,loc='upper center', bbox_to_anchor=(0.0, 1.2),fancybox=True)

        if a.fig_name: 
            plt.savefig(d_plot+'%s' % a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotHourlyStepSTDTime(self,**kwargs):
        ''' Plots standard deviations of histograms of changes every 1 hour, as function of time, must be used with 1 hour time series data.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','colors','fig_name','time_period','include_storms','alpha','labels','ls','xlim','max_val']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        fig = plt.figure(figsize=(10,8))

        STD_df = self.GetHourlyStepSTD(**kwargs)

        ax1 = fig.add_subplot(1,1,1)
        do_y_axis = True
        for _,col_name in enumerate(a.col_names):
            if 'Price' in col_name:
                if do_y_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('$\sigma$ of price')
                    do_y_axis = False
                STD_df.plot(y=col_name,ax=ax2,style=a.ls[_],color=a.colors[_])
            else:
                STD_df.plot(y=col_name,ax=ax1,style=a.ls[_],color=a.colors[_])
        ax1.legend_.remove()
        ax1.set_xlabel('Year')
        ax1.set_ylabel('$\sigma$ of power and consumption')
        # plt.tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
        if a.xlim: 
            ax1.set_xlim(a.xlim)
            try:
                ax2.set_xlim(a.xlim)
            except:
                pass
        ax1.set_ylim([0,a.max_val])
        try:
            ax2.legend_.remove()
            # ax2.set_ylim([0,4*a.max_val])
        # fig.legend(a.labels,loc=2, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fontsize=14) # this creates a combined legend box for both axes
        except:
            pass
        fig.legend(a.labels, loc=3, bbox_to_anchor=(0.2,1), bbox_transform=ax1.transAxes,fontsize=14) # this creates a combined legend box for both axes
        plt.subplots_adjust(top=0.75,bottom=0.1)
        
        if a.fig_name: plt.savefig('../../plots/' + a.fig_name+'.pdf',format='pdf',dpi=500)

    def PlotHourlyStepPercTime(self,**kwargs):
        ''' Plots percentiles of histograms of changes every 1 hour, as function of time, must be used with 1 hour time series data.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['percent','col_names','colors','fig_name','time_period','include_storms','alpha','labels','ls','xlim','max_val']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        fig = plt.figure(figsize=(15,15))

        perc_dict = self.GetHourlyStepPerc(**kwargs)

        # Plot upper percentiles
        ax1 = fig.add_subplot(2,1,1)
        perc_df = perc_dict['upper']
        do_y_axis = True
        for _,col_name in enumerate(a.col_names):
            if 'Price' in col_name:
                if do_y_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('95% percentiles on price')
                    do_y_axis = False
                perc_df.plot(y=col_name,ax=ax2,style=a.ls[_],color=a.colors[_])
            else:
                perc_df.plot(y=col_name,ax=ax1,style=a.ls[_],color=a.colors[_])
        ax1.legend_.remove()
        ax1.set_xlabel('Year')
        ax1.set_ylabel('95% percentiles on power and consumption')
        plt.tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
        if a.xlim: 
            ax1.set_xlim(a.xlim)
            try:
                ax2.set_xlim(a.xlim)
            except:
                pass
        ax1.set_ylim([0,a.max_val])
        try:
            ax2.legend_.remove()
            ax2.set_ylim([0,4*a.max_val])
        except:
            pass
        fig.legend(a.labels, loc='lower right', bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes,fontsize=14) # this creates a combined legend box for both axes

        # Plot lower percentiles
        ax1 = fig.add_subplot(2,1,2)
        perc_df = perc_dict['lower']
        do_y_axis = True
        for _,col_name in enumerate(a.col_names):
            if 'Price' in col_name:
                if do_y_axis:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('5% percentiles on price')
                    do_y_axis = False
                perc_df.plot(y=col_name,ax=ax2,style=a.ls[_],color=a.colors[_])
            else:
                perc_df.plot(y=col_name,ax=ax1,style=a.ls[_],color=a.colors[_])
        ax1.set_xlabel('Year')
        ax1.set_ylabel('5% percentiles on load and consumption')
        ax1.legend_.remove()
        if a.xlim: 
            ax1.set_xlim(a.xlim)
            try:
                ax2.set_xlim(a.xlim)
            except:
                pass
        ax1.set_ylim([-a.max_val,0])
        try:
            ax2.legend_.remove()
            ax2.set_ylim([-4*a.max_val,0])
        except:
            pass
        # ax1.legend(a.labels,loc='lower right')

        plt.subplots_adjust(hspace=0)


        if a.fig_name: plt.savefig('../../plots/' + a.fig_name+'.pdf',format='pdf',dpi=500)

    def PlotDipsVsDuration(self,col_name='',durations=[],min_time_step=0,save=False,d_plot=d_plot):
        power = self.data_df[col_name].values
        N_bins = 1000
        dip_bins = np.linspace(-2,5,N_bins+1)
        dips_histbins = np.zeros([N_bins+1,len(durations)])
        dips_histvalues = np.zeros([N_bins,len(durations)])
        fig = plt.figure(figsize=(15,10))
        ax1 = fig.add_subplot(1,1,1)
        colors = plt.cm.rainbow(np.linspace(0,1,len(durations)))
        popular_dips = np.zeros(len(durations))
        for i,duration in enumerate(durations):
            time_steps_in_bin = int(duration/min_time_step)
            print('Bin with duration of %s min contains %s time steps' % (duration,time_steps_in_bin))
            Power_1 = power[time_steps_in_bin::]
            Power_2 = power[0:-time_steps_in_bin]
            len_Power = len(Power_1)
            dips = Power_1-Power_2
            positive_dips = -dips[dips < 0]
            dips_histvalues[:,i],dips_histbins[:,i] = np.histogram(np.log10(positive_dips),bins=dip_bins)
            # Create fit from only positive histogram values
            y_fit = dips_histvalues[:,i]
            x_fit = dips_histbins[:-1,i]
            x_fit = x_fit[y_fit > 1]
            y_fit = y_fit[y_fit > 1]
            y_fit = np.log10(y_fit)
            fit = np.polyfit(x_fit,y_fit,6)
            if duration <= 1: label = 'Bin duration: %s min' % duration/60
            if duration > 1: label = 'Bin duration: %s hrs' % (duration/60/60)
            ax1.plot(dips_histbins[:-1,i],dips_histvalues[:,i],color=colors[i],label=label,alpha=0.4)
            y_fit = 10.**(np.polyval(fit,x_fit))
            ax1.plot(x_fit,y_fit,color=colors[i])
            # Store most popular dips at this duration
            max_index = list(y_fit).index(max(y_fit))
            popular_dips[i] = x_fit[max_index]
        ax1.set_xlim([-1,5])
        # ax1.set_ylim([1,300])
        ax1.set_xlabel('log(dip in Power) [MW]')
        ax1.set_ylabel('Histogram')
        # ax1.set_ylabel('Normalized histogram [%]')
        ax1.set_yscale('log')
        plt.legend()
        if save: plt.savefig(d_plot+'dip_histograms',dpi=200)

        fig = plt.figure(figsize=(15,10))
        ax1 = fig.add_subplot(1,1,1)
        ax1.pcolormesh(durations/60/60,dip_bins,dips_histvalues, cmap='viridis')
        ax1.set_ylabel('log($\Delta$ P$_{dip}$) [MW]')
        ax1.set_xlabel('Duration [hrs]')
        durations_center = np.array([(durations[i+1]+durations[i])/2. for i in range(len(durations)-1)])
        durations_center = np.insert(durations_center,0,0)
        ax1.plot(durations/60/60,popular_dips,'x',color='w',mew=2,ms=10,label='Most popular dips for this duration')
        ax1.set_ylim(bottom=0)
        plt.legend()
        ax2 = ax1.twiny()
        ax2.set_xlabel('Duration [min]')
        ax2.set_xticks(durations/60/60)
        ax2.set_xticklabels([str(duration) for duration in durations],fontsize=8) 
        if save: plt.savefig(d_plot+'dips_vs_duration',dpi=300)

    def PlotDurationCurves(self,col_names=[],min_time_step=0,colors=[],labels=[]):

        N_bins = 1000
        duration_curve = np.zeros(N_bins)
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,1,1)
        for col_name,color,label in zip(col_names,colors,labels):
            power = self.data_df[col_name].values
            power_bins = np.linspace(np.min(power),np.max(power),N_bins+1)
            duration = [len(power[(power >= power_bins[_]) & (power > power_bins[_+1])])*min_time_step/60/60 for _ in range(len(power_bins)-1)]# hours
            power_bins_center = power_bins[0:-1] + (power_bins[1]-power_bins[0])/2.
            ax1.plot(duration,power_bins_center,color,lw=3,label=label)
        ax1.set_xlabel('[Hours]')
        ax1.set_ylabel('[MW]')
        plt.legend()

    def PlotCarpets(self,**kwargs):
        ''' Make "carpet plot" of hourly ramps on day vs. time
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','years','fig_name','max_step']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df = self.data_df.copy()

        num_plots = len(a.col_names)
        fig, axs = plt.subplots(num_plots,1,figsize=(16/num_plots,15))
        cmap = plt.get_cmap('jet')
        
        i = 0

        images = []
        for col_name,year in zip(a.col_names,a.years):

            # Get hourly ramps over the chosen time period
            # hourly_steps = getattr(self,'hourly_steps_'+col_name)
            if 'hourly_steps_'+col_name not in data_df.keys(): 
                self.AddHourlySteps(col_name=col_name,scale_by_capacity=True,**kwargs)
            data_df = self.data_df.copy()
            hourly_steps = data_df['hourly_steps_'+col_name].values
            datetimes = data_df['datetime'].values
                
            if year == 'mean':
                mean_carpet = np.zeros([365,24])
                years = np.unique(np.array([_.astype('datetime64[D]').item().year for _ in datetimes]))
                N_years = len(years)
                for year1 in years:
                    time_period = [dt.datetime(year1, 1, 1, 1, 0, 0),dt.datetime(year1+1, 1, 1, 1, 0, 0)]
                    time_period_ind = [(np.datetime64(time_period[0]) <= datetimes) & (datetimes < np.datetime64(time_period[1]))]
                    time_period_ind = np.array(time_period_ind)[0]
                    hourly_steps_temp = hourly_steps[time_period_ind]
                    # Reshape for plotting
                    if len(hourly_steps_temp) == 8784: 
                        hourly_steps_temp = hourly_steps_temp[0:8760]
                    # if len(hourly_steps_temp) == 8761: 
                    #     dt1 = datetimes[time_period_ind]
                    #     print(min(dt1))
                    #     print(max(dt1))
                    if len(hourly_steps_temp) != 8760: 
                        print('Year %s has %s datapoints (not 8760), will not be used' % (year1,len(hourly_steps_temp)))
                        N_years -= 1
                    if len(hourly_steps_temp) == 8760: 
                        print('Year %s has enough datapoints' % (year1))
                        hourly_steps_temp = hourly_steps_temp.reshape(365,24)
                        mean_carpet = mean_carpet + hourly_steps_temp
                hourly_steps_carpet = mean_carpet / N_years

            if year != 'mean':
                time_period = [dt.datetime(year, 1, 1, 1, 0, 0),dt.datetime(year+1, 1, 1, 1, 0, 0)]
                time_period_ind = [(np.datetime64(time_period[0]) <= datetimes) & (datetimes <= np.datetime64(time_period[1]))]
                time_period_ind = np.array(time_period_ind)[0]
                hourly_steps_temp = hourly_steps[time_period_ind]
                # Reshape for plotting
                if len(hourly_steps_temp == 8784): hourly_steps_temp = hourly_steps_temp[0:8760]
                hourly_steps_carpet = hourly_steps_temp.reshape(365,24)

            if a.max_step == False:
                max_step = max(abs(hourly_steps_carpet))
            else:
                max_step = a.max_step

            days = np.arange(366)
            hours = np.arange(25)
            ax1 = axs[i]
            i += 1
            # ax1 = fig.add_subplot(num_plots, 1, i)
            images.append(ax1.imshow(hourly_steps_carpet,cmap=cmap))
            # im = ax1.pcolormesh(hours, days, hourly_steps_carpet, cmap=cmap, vmin=-max_step, vmax=max_step)#, norm=norm)
            # ax1.set_title(aux.pretty_label(col_name))
            if i == 3: ax1.set_xlabel('Hours in a day')
            ax1.set_ylabel('Months in year')
            ax1.set_yticks(np.arange(12)*30+15)
            ax1.set_yticklabels([str(_+1) for _ in range(12)])
            # ax1.set_xticks(np.arange(12)*2+2)
            # ax1.set_xticklabels([str(int(_*2.+2)) for _ in range(12)])
            ax1.invert_yaxis()
            ax1.set_aspect('auto')
            # if i < len(a.col_names): fig.colorbar(im)
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())

        for im in images:
            im.callbacksSM.connect('changed', update)

        # Add colorbar above
        plt.subplots_adjust(top=0.88,bottom=0.05)
        cax = plt.axes([0.15, 0.9, 0.75, 0.02])
        cb = plt.colorbar(images[0], cax=cax, orientation='horizontal', ticklocation = 'top')
        cb.set_label(r'Hourly ramps [% of installed capacity]', labelpad=10)


        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf',format='pdf',dpi=300)

    def PlotPie(self,**kwargs):
        ''' Make pie chart of integrated power fractions.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['int_power','colors','labels','add','legend','alpha']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add:
            fig = plt.gcf()
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1,1,1,polar=True)

        start_at            =   np.pi/2.
        int_power           =   np.flip(a.int_power,axis=0)
        int_power_perc      =   int_power/np.sum(int_power)*100.
        widths              =   int_power*2*np.pi/np.sum(int_power)
        radiis              =   np.array([1]*len(int_power)) # scaled to one
        thetas              =   np.append(start_at,np.add.accumulate(widths[0:-1])+start_at) # radians
        xlabels = ['%.1f %%' % (int_power_perc[_]) for _ in range(len(int_power))]
        # xlabels = ['%.1f %%\n         %.1e MWh' % (int_power_perc[_],int_power[_]) for _ in range(len(int_power))]
        # xlabels[2] = '%.1f %%\n %.1e MWh' % (int_power_perc[2],int_power[2]) # fix text on left-hand side of pie
        theta_text = thetas+widths/2.
        # theta_text[2] = theta_text[2]-np.pi/8
        radiis_text = radiis+0.12
        # radiis_text[2] += 0.1
        [ax1.text(x,y,s,fontsize=18,color=c,horizontalalignment='center',verticalalignment='center') for x,y,s,c in zip(theta_text,radiis_text,xlabels,a.colors[::-1])]
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        for _ in range(len(thetas)):
            bar = ax1.bar(thetas[_]+widths[_]/2., 1., width=widths[_], bottom=0.0, label=a.labels[::-1][_])
            bar[0].set_alpha(a.alpha)
            bar[0].set_facecolor( a.colors[::-1][_] )
        # Axes and legend
        # h1,l1 = ax1.get_legend_handles_labels()
        # if a.legend: ax1.legend(handles=h1[::-1], labels=a.labels, fontsize=15, loc=[1.2,0.2])
        ax1.spines['polar'].set_visible(False)

    def PlotHisto(self,**kwargs):
        ''' Plot histogram of an attribute, a column in the dataframe or a supplied data array.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['histogram_values','epoch','fig_name','bins','max_val','log','ylog','power_name',\
            'xlog','labels','colors','ls','alpha','xlim','xlab','ylab','add','data','remove_zeros','legend']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        if a.add:
            fig = plt.gcf()
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1,1,1)
            if a.xlab:
                ax1.set_xlabel(a.xlab)
            else:
                ax1.set_xlabel('Values')
            # if a.ylab:
            #     ax1.set_ylabel(a.ylab)
            # else:
            ax1.set_ylabel('Percentage')

        for _,histogram_value in enumerate(a.histogram_values):
            try:
                array = getattr(self,histogram_value)
            except:
                try:
                    array = data_df[histogram_value].values
                except:
                    print('Could not find values for %s, will look for supplied data' % histogram_value)
                    try:
                        array = a.data
                    except ValueError:
                        print('No data supplied')

            # if a.epoch != 2:
            #     array = array[data_df['low_penetration_index_'+ a.power_name] == a.epoch]

            if a.max_val:
                array = array[array <= a.max_val]

            if a.log:
                array = np.log10(abs(array[abs(array) > 0]))

            if a.xlim:
                array = array[(array >= a.xlim[0]) & (array <= a.xlim[1])]

            if a.bins == 'integer':
                a.bins = int(np.max(array)-np.min(array))

            hist, bins = np.histogram(array, bins=a.bins)
            delta_bin = bins[1]-bins[0]
            hist = np.append(0,hist)
            hist = hist*100/np.sum(hist)
            bins_center = np.append(bins[0]-delta_bin,bins[0:-1])+delta_bin/2.
            if a.remove_zeros:
                bins_center = bins_center[hist != 0]
                hist = hist[hist != 0]
            # Add zeros:
            hist = np.append(0,hist)
            hist = np.append(hist,0)
            bins_center = np.append(bins_center[0]-delta_bin,bins_center)
            bins_center = np.append(bins_center,bins_center[-1]+delta_bin)
            hist = hist*100/np.trapz(hist,x=bins_center)
            # hist = hist*100/np.sum(hist)

            if a.labels:
                label = a.labels[_]
            else:
                try:
                    label = aux.pretty_label(histogram_value)
                except:
                    label = histogram_value
            ax1.plot(bins_center,hist,color=a.colors[_],ls=a.ls[_],drawstyle='steps',alpha=a.alpha,label=label)
        if a.legend: plt.legend(fontsize=13)
        if a.ylog:
            ax1.set_yscale('log')
        if a.xlog:
            ax1.set_xscale('log')
        if a.xlim: ax1.set_xlim(a.xlim)
        ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    def PlotYearlyData(self,**kwargs):
        ''' Plots data aggregated for one year at a time.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','colors','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if not hasattr(self,'yearly_data'): self.AddYearlyData()

        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Integrated power [TWh/y]')

        for i,col_name in enumerate(a.col_names):
            ax1.plot(self.yearly_data['years'].values,self.yearly_data[col_name].values/1e6,color=a.colors[i],ls='--')
            ax1.plot(self.yearly_data['years'].values,self.yearly_data[col_name].values/1e6,color=a.colors[i],marker='o',lw=0,ms=10,label=col_name)
        plt.legend()

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotPenetration(self,**kwargs):
        ''' Plots penetration in percent of total consumption.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','load_name','colors','fig_name','yearly','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        self.AddPenetrationFraction(power_names=[a.power_name],load_names=[a.load_name],yearly=a.yearly)
        alpha = getattr(self,'alpha')
        print(alpha)

        years = alpha['years'].astype(int)
        alpha = alpha.drop('years',axis=1)
        print(alpha)

        fig = plt.figure(figsize=(8,7))
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel('Year')
        ax1.set_ylabel(r'$\alpha_{\mathrm{renewables}}$ [%]')

        for i,power_name in enumerate(alpha.keys()):
            print(power_name)
            alphas = alpha[power_name].values
            ax1.plot(years[alphas > 0],alphas[alphas > 0],color=a.colors[i],ls='--')
            ax1.plot(years[alphas > 0],alphas[alphas > 0],color=a.colors[i],marker='o',lw=0,ms=10,label=aux.pretty_label(power_name))
        plt.legend(fontsize=14)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=300)

    def PlotTimeSeries(self,**kwargs):
        ''' Plots time series of data for specific time period (all times by default).
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','labels','ylim','time_cut','add','ylab','legend','alpha','colors','ls','fig_name','add','two_axes','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add:
            fig = plt.gcf()
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(1,1,1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel(a.ylab)

        data_df             =   self.data_df.copy()
        datetimes           =   data_df['datetime'].values

        if not a.labels:
            a.labels = [aux.pretty_label(col_name) for col_name in a.col_names]

        if a.time_cut:
            mask = np.array([(datetimes > a.time_cut[0]) & (datetimes < a.time_cut[1])])[0]
            data_df = data_df[mask]

        if len(a.ls) != len(a.col_names):
            print('List of line styles not the same length as list of column names, will use solid lines')
            a.ls = ['-']*len(a.col_names)

        if a.two_axes:
            do_y_axis = True
            for col_name,color,label,ls in zip(a.col_names,a.colors,a.labels,a.ls):
                data = data_df[col_name].values
                time = data_df['datetime']
                if 'Price' in col_name:
                    if do_y_axis:
                        ax2 = ax1.twinx()
                        ax2.set_ylabel('Price [EUR]')
                        do_y_axis = False
                    ax2.plot(time,data.real,color,ls=ls,label=label,alpha=a.alpha)
                else:
                    ax1.plot(time,data.real,color,ls=ls,label=label,alpha=a.alpha)
        else:
            for col_name,color,label,ls in zip(a.col_names,a.colors,a.labels,a.ls):
                data = data_df[col_name].values
                time = data_df['datetime']
                ax1.plot(time,data.real,color,ls=ls,label=label,alpha=a.alpha)

        if a.ylim: 
            ax1.set_ylim(a.ylim)

        if a.legend: fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fancybox=True,fontsize=14) # this creates a combined legend box for both axes

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=500)

    def HighlightDips(self,**kwargs):
        ''' Highlights identified dips on existing time series plot.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','time_cut','alpha','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        if time_cut:
            datetime = data_df['datetime'].values
            mask = [(data_df['datetime'] > time_cut[0]) & (data_df['datetime'] < time_cut[1])][0]
            data_df = data_df.loc[mask,:] #& (data_df['datetime'] < times[1])]

        # For the plotting
        ax1 = plt.gca()
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        colors = ['r','b']
        y_start = [0.5,0]

        try:
            print(a.power_name+'_dip_starts')
            all_dip_starts = getattr(self,a.power_name+'_dip_starts')
            all_dip_mins = getattr(self,a.power_name+'_dip_mins')
            all_magnitudes = getattr(self,a.power_name+'_magnitudes')
            freq_indicator = getattr(self,a.power_name+'_freq_indicator')
        except:
            print('Dips have not been calculated yet')

        for i,freq in enumerate(['low','high']):

            # Select low or high freq dips
            dip_starts = all_dip_starts[freq_indicator == i]
            dip_mins = all_dip_mins[freq_indicator == i]
            magnitudes = all_magnitudes[freq_indicator == i]

            if a.time_cut:
                dip_mins = dip_mins[(dip_starts > time_cut[0]) & (dip_starts < time_cut[1])]
                magnitudes = magnitudes[(dip_starts > time_cut[0]) & (dip_starts < time_cut[1])]
                dip_starts = dip_starts[(dip_starts > time_cut[0]) & (dip_starts < time_cut[1])]

            print('Found these dip magnitudes:')
            print(magnitudes)

            if len(dip_mins) > 0:
                # Check that we start with a dip_start
                if dip_starts[0] > dip_mins[0]:
                    dip_mins = dip_mins[1::]
                if len(dip_starts) > len(dip_mins):
                    dip_starts = dip_starts[0:len(dip_mins)]
                N_dips = len(dip_starts)
                for dip_start,dip_min in zip(dip_starts,dip_mins):
                    # if dip_start == dip_min: dip_min += 1
                    ax1.fill_between([dip_start,dip_min],y_start[i],y_start[i]+0.5,\
                        facecolor=colors[i],alpha=a.alpha, transform=trans) 

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def HighlightPenetration(self,**kwargs):
        ''' Highlights identified times of high/low penetration fraction on existing time series plot.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','time_cut','colors','alpha','fig_name','alpha_cuts']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        if a.time_cut:
            datetime = data_df['datetime'].values
            mask = [(data_df['datetime'] > a.time_cut[0]) & (data_df['datetime'] < a.time_cut[1])][0]
            data_df = data_df.loc[mask,:] #& (data_df['datetime'] < times[1])]
        data_df = data_df.reset_index(drop=True)
        datetimes = data_df['datetime'].values
        high_penetration_index = data_df['high_penetration_index_'+a.power_name].values
        low_penetration_index = data_df['low_penetration_index_'+a.power_name].values

        # Interpolate to finer resolution (otherwise, fill_between will not show the shortest periods)
        new_datetimes = np.arange(np.min(datetimes),np.max(datetimes),np.timedelta64(30,'m'))
        hours = (datetimes - np.min(datetimes))/np.timedelta64(1,'h')
        new_hours = np.arange(0,np.max(hours),0.5)

        interp_func = interp1d(hours,high_penetration_index)
        new_high_penetration_index = interp_func(new_hours)
        interp_func = interp1d(hours,low_penetration_index)
        new_low_penetration_index = interp_func(new_hours)

        # For the plotting
        ax1 = plt.gca()
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        colors = ['r','b']
        if a.colors: colors=a.colors
        ylim = ax1.get_ylim()

        # print(high_penetration_index)
        # print(new_high_penetration_index)

        ax1.fill_between(new_datetimes, ylim[0], ylim[1], where=new_high_penetration_index > 0, facecolor=colors[0], alpha=a.alpha, interpolate = False)
        ax1.fill_between(new_datetimes, ylim[0], ylim[1], where=new_low_penetration_index > 0, facecolor=colors[1], alpha=a.alpha, interpolate = False)
        if a.alpha_cuts:
            ax1.plot(datetimes,np.zeros(len(datetimes))+a.alpha_cuts[0],color='grey',ls='--',lw=3)
            ax1.plot(datetimes,np.zeros(len(datetimes))+a.alpha_cuts[1],color='grey',ls='--',lw=3)
        ax1.set_ylim(ylim)

    def PlotMagnitudeVsDuration(self,**kwargs):
        ''' Plots magnitudes vs durations of identified ramps as countour plot.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','magnitude_range','duration_range','times','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        magnitudes = getattr(self,a.power_name+'_SDA_ramp_magnitudes')
        durations = getattr(self,a.power_name+'_SDA_ramp_durations_min')
        N_ramps = len(magnitudes)

        print('Found %s ramps' % (len(magnitudes)))
        print('Found %s ramps' % (len(durations)))
        # magnitudes = np.array([])
        # durations = np.array([])
        # if a.times:
        #     magnitudes = np.append(magnitudes,getattr(self,a.power_name+'_magnitudes_'+times))
        #     durations = np.append(durations,getattr(self,a.power_name+'_durations_'+times))
        # else:
        #     magnitudes = np.append(magnitudes,getattr(self,a.power_name+'_magnitudes'))
        #     durations = np.append(durations,getattr(self,a.power_name+'_durations'))           

        sns.set(style="white", color_codes=True)

        magnitudes = magnitudes[(durations > a.duration_range[0]) & (durations < a.duration_range[1])]
        durations = durations[(durations > a.duration_range[0]) & (durations < a.duration_range[1])]
        durations = durations[(magnitudes > a.magnitude_range[0]) & (magnitudes < a.magnitude_range[1])]
        magnitudes = magnitudes[(magnitudes > a.magnitude_range[0]) & (magnitudes < a.magnitude_range[1])]

        print('Cutting out %s out of %s dips for being too long/deep' % (N_ramps - len(magnitudes),N_ramps))

        plot = (sns.jointplot(x=durations, y=magnitudes, kind='kde', color="skyblue",height=8)
                    .set_axis_labels("Duration [min]", "Magnitude [MW]"))

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotiFFT(self,**kwargs):
        ''' Plots Fast Fourier Transform (FFT) for wind power.
        '''


        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','freq_cuts','labels','ylim','legend','ylab','fig_name','new_y_axis','add','time_cut','colors','test']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add:
            ax0 = plt.gca()
            if a.new_y_axis:
                ax1 = ax0.twinx()
                if a.ylab: ax1.set_ylabel(a.ylab)
            else:
                ax1 = ax0
        else:
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(1,1,1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Power [MW]')
            if a.ylab: ax1.set_ylabel(a.ylab)

        data_df             =   self.data_df.copy()
        power               =   data_df[a.power_name].values
        mean_power          =   np.mean(power[power != -1])
        if a.time_cut:
            mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
            data_df = data_df[mask]
        datetime            =   data_df.datetime
        ax1.plot(datetime,data_df[a.power_name].values,color='k',ls='--',label='Residual Load')

        if a.colors == 'rainbow':
            colors = cm.rainbow(np.linspace(0, 1, len(a.freq_cuts)))
        else:
            colors = a.colors

        iFFT_sum = 0 
        # if len(a.freq_cuts) > 1:
        for cut,freq_cut in enumerate(a.freq_cuts):
            self.AddiFFT(freq_cut=freq_cut,power_name=a.power_name,verbose=False,test=a.test)
            data_df             =   self.data_df.copy()
            if a.time_cut:
                mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
                data_df = data_df[mask]
            iFFT                =   data_df[a.power_name+'_iFFT'].values.real
            if a.test: iFFT                =   data_df[a.test+'_iFFT'].values.real
            iFFT_sum += iFFT
            datetime            =   data_df.datetime
            if a.labels: 
                label=a.labels[cut]
            else:
                label='iFFT for freq cut # %s' % str(cut)
            ax1.plot(datetime,iFFT,color=colors[cut],label=label)
        # ax1.plot(datetime,iFFT_sum + mean_power,color='magenta',ls=':',label='iDFTs summed')
        
        # For comparison
        self.AddiFFT(power_name=a.power_name,test=a.test)
        data_df             =   self.data_df.copy()
        if a.time_cut:
            mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
            data_df = data_df[mask]
        iFFT                =   data_df[a.power_name+'_iFFT'].values.real
        if a.test: iFFT                =   data_df[a.test+'_iFFT'].values.real
        datetime            =   data_df.datetime
        # ax1.plot(datetime,iFFT,color=a.colors[0],ls=':',label='iFFT')

        ax1.plot(datetime,np.zeros(len(datetime)),color='k',ls='--',lw=1)
        ax1.grid()
        # if a.mark_days:
        #     days = [datetime[_].day for _ in range(len(datetime))]
        #     for day in days:
        #         ax1.plot([day,np.datetime64()],np.zeros(len(datetime)),color='k',ls='--',lw=1)
        if a.ylim: 
            ax1.set_ylim(a.ylim)
        if a.legend: 
            plt.legend(fontsize=13)
            # plt.legend(loc='center right')
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    def PlotFFT(self,**kwargs):
        ''' Plots Fast Fourier Transform (FFT) for wind power.
        '''


        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','freq_cuts','labels','ylim','legend','ylab','fig_name','new_y_axis','add','time_cut','colors']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add:
            ax1 = plt.gca()
            if a.new_y_axis:
                ax1 = ax1.twinx()
        else:
            fig = plt.figure(figsize=(15,10))
            ax1 = fig.add_subplot(1,1,1)
            ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel('FFT (real part)')
            if a.ylab: ax1.set_ylabel(a.ylab)

        data_df             =   self.data_df.copy()
        if a.power_name+'_FFT' not in data_df.keys():
            self.AddFFT(power_name=a.power_name,verbose=False)        
        data_df             =   self.data_df.copy()
        freq                =   data_df[a.power_name+'_freq'].values
        FFT                 =   data_df[a.power_name+'_FFT'].values

        ax1.plot(freq,FFT,color='r',label='FFT of %s' % a.power_name)

        # ax1.set_xscale('log')
        # ax1.set_yscale('log') 

        if a.ylim: 
            ax1.set_ylim(a.ylim)
        if a.legend: plt.legend()
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.pdf', format='pdf', dpi=500)

    ### Data handling

    def GetFluctuations(self,**kwargs):
        ''' Calculates durations and integrated energy of fluctuations within given epoch, returns in dictionary

        Parameters
        ----------
        cons_ob: object
            Object containing data of consumption, default: ''
        epoch: int
            Defines which epoch we're looking at (0 for low wind penetration, 1 for high)

        duration_cuts: int/float
            The durations [hours] seperating different fluctuations, default: [[0,5],[5,10],[10,50]]
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['cons_ob','epoch','col_name','power_name','freq_cuts','alpha_cuts','test']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.epoch == 1: print('\n Now calculating fluctuations during epochs of high penetration')
        if a.epoch == 0: print('\n Now calculating fluctuations during epochs of low penetration')
        if a.epoch == 2: print('\n Now calculating fluctuations during all epochs!')

        data_df = self.data_df.copy()
        datetimes = data_df['datetime'].values

        # Select object to get penetration fraction from
        if a.cons_ob == '':
            a.cons_ob = self
        if a.epoch != 2:
            if 'high_penetration_index_'+a.power_name not in a.cons_ob.data_df.keys():
                a.cons_ob.AddHighPenetrationIndex(alpha_cuts=a.alpha_cuts,power_name=a.power_name)
            else:
                print('High/low penetration already calculated!')

        if a.epoch == 2: 
            int_power_dict      =   {}
            ramps_dict          =   {}
            total_int_power     =   0
            for cut,freq_cut in enumerate(a.freq_cuts):
                self.AddiFFT(freq_cut=freq_cut,power_name=a.col_name,verbose=False,test=a.test)
                data_df             =   self.data_df.copy()
                iFFT                =   data_df[a.col_name+'_iFFT'].values.real
                if a.test: 
                    iFFT = data_df[a.test+'_iFFT'].values.real
                hours               =   (data_df.datetime-data_df.datetime[0])/np.timedelta64(1,'h')
                # Measure fluctuations for all epochs
                datetime            =   self.data_df.datetime
                hours               =   (datetime-np.min(datetime))/np.timedelta64(1,'h')
                # Integrate power
                int_power           =   np.trapz(np.abs(iFFT),hours)/(len(hours)/(365*24.)) # annual value
                ramps               =   np.array([iFFT[_+1] - iFFT[_] for _ in range(len(iFFT)-1)])
                total_int_power     +=   np.sum(int_power)
                int_power_dict['cut'+str(cut)]  =   int_power
                ramps_dict['cut'+str(cut)]      =   ramps
        else:
            if a.epoch == 0: 
                start_times,end_times = getattr(a.cons_ob,'low_penetration_start_times'),getattr(a.cons_ob,'low_penetration_end_times')
            if a.epoch == 1: 
                start_times,end_times = getattr(a.cons_ob,'high_penetration_start_times'),getattr(a.cons_ob,'high_penetration_end_times')
            int_power_dict      =   {}
            durations_dict      =   {}
            ramps_dict          =   {}
            total_int_power     =   0
            for cut,freq_cut in enumerate(a.freq_cuts):
                self.AddiFFT(freq_cut=freq_cut,power_name=a.col_name,verbose=False,test=a.test)
                data_df             =   self.data_df.copy()
                iFFT                =   data_df[a.col_name+'_iFFT'].values.real
                if a.test: iFFT = data_df[a.test+'_iFFT'].values.real
                hours               =   (data_df.datetime-data_df.datetime[0])/np.timedelta64(1,'h')
                # Measure fluctuations for all epochs
                datetime            =   self.data_df.datetime
                int_power           =   np.zeros(len(start_times))
                durations           =   np.zeros(len(start_times))
                ramps               =   np.array([])
                _                   =   0
                for start_time,end_time in zip(start_times,end_times):
                    # Cut out FFT spectrum
                    mask = np.array([(datetime >= start_time) & (datetime <= end_time)])[0]
                    iFFT_cut            =   iFFT[mask]
                    if len(iFFT_cut) > 0:
                        hours               =   (datetime[mask]-start_time)/np.timedelta64(1,'h')
                        # Integrate power
                        # iFFT_cut            =   iFFT_cut - np.mean(iFFT_cut) # TEST, DOES THIS MAKE SENSE????
                        int_power[_]        =   np.trapz(np.abs(iFFT_cut),hours)/(len(hours)/(365*24.)) # annual value
                        # int_power[_]            =   np.trapz(np.abs(iFFT_cut),hours)
                        # print(iFFT_cut)
                        # print(start_time,end_time)
                        # print(len(iFFT_cut))
                        # print(int_power[_])
                        # a = asfa 
                        durations[_]        =   0
                        ramps_temp          =   np.array([iFFT_cut[_+1] - iFFT_cut[_] for _ in range(len(iFFT_cut)-1)])
                        ramps               =   np.append(ramps,ramps_temp)
                        _                   +=  1
                int_power               =   int_power[0:_-2]

                total_int_power                     +=   np.sum(int_power)
                int_power_dict['cut'+str(cut)]      =   int_power
                durations_dict['cut'+str(cut)]      =   durations
                ramps_dict['cut'+str(cut)]      =   ramps

        print('Relative amount of energy in each frequency interval:')
        for cut in range(cut+1):
            print('For frequency interval: %.2e to %.2e Hz: %.2e MWh' % (a.freq_cuts[cut][0],a.freq_cuts[cut][1],np.sum(int_power_dict['cut'+str(cut)])))
            print('%.1f %% of integrated energy across all frequencies' % (np.sum(int_power_dict['cut'+str(cut)])/total_int_power*100.))


        print('------')

        fluctuations        =   dict(int_power=int_power_dict,ramps=ramps_dict)

        return(fluctuations)

    def PrintIntPowerAsTable(self,freq_cuts,fluctuations):


        file_name = "../../data/tables/int_power_MWh.txt"
        file = open(file_name,"w")

        unit = 1e6 # Wh

        fluc1,fluc2,fluc3,fluc4 = fluctuations[0],fluctuations[1],fluctuations[2],fluctuations[3]

        text = ['$< 10$ hrs      &       ',\
                    '10 - 24 hrs     &       ',\
                    '1 - 7 d         &       ',\
                    '7 d - 3 mos.    &       ',\
                    '3 mos. - 1 yr   &       ',\
                    '$> 1$ yr        &       ']

        for _,freq_cut in enumerate(freq_cuts):

            try: 
                file.write(text[_]+'%.2f & %.2f & %.2f & %.2f \\\\ \n' % (fluc1['int_power']['cut'+str(int(_))]/unit,fluc2['int_power']['cut'+str(int(_))]/unit,fluc3['int_power']['cut'+str(int(_))]/unit,fluc4['int_power']['cut'+str(int(_))]/unit))
            except:
                file.write(text[_]+'%.2f & %.2f & %.2f & - \\\\ \n' % (fluc1['int_power']['cut'+str(int(_))]/unit,fluc2['int_power']['cut'+str(int(_))]/unit,fluc3['int_power']['cut'+str(int(_))]/unit))


        file.close()

    def ExtremeHourlyRamps(self,**kwargs):
        ''' Finds percentiles and extreme hourly ramps with and without storms, and stores as tex table.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','percent']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        perc_table = [0]*len(a.col_names)
        for _,col_name in enumerate(a.col_names):
            perc_table[_] = col_name
            for include_storms in [True,False]:
                perc_dict = self.GetHourlyStepPerc(col_names=[col_name],include_storms=include_storms,percent=a.percent,time_period='year',verbose=False)
                perc_lower = np.array([_ for _ in perc_dict['lower'][col_name]])
                perc_upper = np.array([_ for _ in perc_dict['upper'][col_name]])
                perc_table[_] = perc_table[_] + ' & %.1f & %.1f \\%%' % (np.min(perc_lower),np.max(perc_upper))
            perc_table[_] = perc_table[_] + ' \\\\ \n'

        file_name = "../../data/tables/extreme_hourly_ramps_perc.txt"
        file = open(file_name,"w")
        file.writelines(perc_table)
        file.close()

        extrema_table = [0]*len(a.col_names)
        for _,col_name in enumerate(a.col_names):
            extrema_table[_] = col_name
            for include_storms in [True,False]:
                self.AddHourlySteps(col_name=col_name,include_storms=include_storms,scale_by_capacity=False,verbose=False)
                data_df = self.data_df.copy()
                hourly_ramps = data_df['hourly_raw_steps_'+col_name]
                extrema_table[_] = extrema_table[_] + ' & %.1f\\%%' % (np.min(hourly_ramps))
                extrema_table[_] = extrema_table[_] + ' & %.1f\\%%' % (np.max(hourly_ramps))
            extrema_table[_] = extrema_table[_] + ' \\\\ \n'

        file = open("../../data/tables/extreme_hourly_ramps_extrema.txt","w")
        file.writelines(extrema_table)
        file.close()

        print('Extreme ramps have been written to file %s!' % file_name)

    def MagnitudesToRamps(self,magnitudes,times,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','time_cut','d_data','duration_ramp']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        # Load capacities
        capacity_ob = cap.Capacity(file_path=a.d_data+'ENS/',file_name='capacity_df')
        capacity_ob.AddData(file_path=a.d_data+'ENS/')
        data_df = capacity_ob.data_df.copy()

        if 'DK1' in a.col_name: capacity_df = data_df[data_df['communenr'] > 400].copy()
        if 'DK2' in a.col_name: capacity_df = data_df[data_df['communenr'] <= 400].copy()
        if 'BO' in a.col_name: capacity_df = data_df[data_df['communenr'] == 400].copy()
        if a.col_name == 'TotalWindPower': capacity_df = data_df.copy() # capacity,date = capacity_ob.data_df['capacity'],capacity_ob.data_df['date'].values
        if a.col_name == 'OnshoreWindPower': capacity_df = data_df[data_df['placing'] == 'LAND'].copy() 
        if a.col_name == 'OffshoreWindPower': capacity_df = data_df[data_df['placing'] == 'HAV'].copy()
        capacity_df['accum_cap'] = np.cumsum(capacity_df['capacity'].values)

        # Merge capacities into data
        data_df = self.data_df[[a.col_name,'datetime']].copy()
        data_df_w_capacity = pd.merge(data_df,capacity_df,on='datetime',how='outer')#.set_index('datetime')
        # remove any duplicates in time
        data_df_w_capacity = data_df_w_capacity.drop_duplicates(subset='datetime')
        data_df_w_capacity = data_df_w_capacity.sort_values('datetime').reset_index(drop=True)
        # fill missing values
        data_df_w_capacity['accum_cap'] = data_df_w_capacity['accum_cap'].ffill() # Continue with last measured total capacity
        data_df_w_capacity[a.col_name] = data_df_w_capacity[a.col_name].ffill()
        data_df_w_capacity[a.col_name] = data_df_w_capacity[a.col_name].bfill()

        # Cut time out?
        if a.time_cut:
            time_cut_ind = [(np.datetime64(a.time_cut[0]) <= data_df_w_capacity['datetime']) & (data_df_w_capacity['datetime'] <= np.datetime64(a.time_cut[1]))]
            time_cut_ind = np.array(time_cut_ind)[0]
            data_df_w_capacity = data_df_w_capacity.loc[time_cut_ind,:].reset_index()

        # Extract capacities at relevant times:
        capacity            =   data_df_w_capacity['accum_cap'].values
        datetimes           =   data_df_w_capacity['datetime']
        print('Finding installed capacity at %s ramp times ' % (len(times)))
        capacity_cut        =   np.array([capacity[datetimes == time] for time in times])
        print('done!')

        # Convert magnitudes to actual ramps
        ramps = magnitudes/capacity_cut*100.
        print(len(ramps))
        # ramps = np.nan_to_num(ramps)

        if not a.time_cut:
            # Take out the first ramp value (when production drops from 0 to some value...)
            ramps = ramps[ramps != 0]
            # ramps = ramps[1::]
            # Save in this object for later if they are complete
            setattr(self,'ramps_'+a.col_name,ramps)

    def GetSDA(self,**kwargs):
        ''' Returns SDA result as numpy array.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['SDAtype','power_name','epoch']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        datetimes = data_df['datetime'].values

        times_ramps         =   getattr(self,a.power_name+'_SDA_times_ramps')
        high_penetration_index = data_df['high_penetration_index_'+a.power_name]
        times_epoch         =   datetimes[high_penetration_index == a.epoch]

        ramps_found         =   np.array([])
        for time in times_epoch:
            # print(time)
            index = list(np.where(times_ramps == time)[0])
            if len(index) > 0:
                # print(time)
                ramps_found         =   np.append(ramps_found,index)
        ramps_found = ramps_found.astype(int)

        if a.SDAtype == 'duration': SDA_result = getattr(self,a.power_name+'_SDA_ramp_durations_min')
        if a.SDAtype == 'magnitude': SDA_result = getattr(self,a.power_name+'_SDA_ramp_magnitudes')

        print('Found %s ramps starting in this epoch' % len(ramps_found))

        SDA_result_cut = np.array([SDA_result[_] for _ in ramps_found])

        return(SDA_result_cut)

    def GetHourlyStepSTD(self,**kwargs):
        ''' Calculates Standard Deviations of step sizes, using result from AddHourlySteps(), over specified bin size.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','time_period','time_cut']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df = self.data_df.copy()
        datetime = data_df.datetime.values
        if a.time_cut:
            mask = np.array([(datetime >= np.datetime64(a.time_cut[0])) & (datetime <= np.datetime64(a.time_cut[1]))])[0]
            data_df = data_df[mask]

        # Get all hourly steps
        for col_name in a.col_names:
            self.AddHourlySteps(col_name=col_name,include_storms=a.include_storms,scale_by_capacity=True)
        data_df = self.data_df.copy()

        # Identify individual years
        datetime = data_df.datetime.values
        if a.time_period =='all':
            STD = np.zeros(1)
        if a.time_period =='year':
            min_year = min(datetime).astype('datetime64[D]').item().year
            max_year = max(datetime).astype('datetime64[D]').item().year+1
            time_periods = [[np.datetime64('%s-01-01' % year),np.datetime64('%s-01-01' % (year+1))] for year in np.arange(min_year,max_year)]
            STD = np.zeros(len(time_periods))
        if a.time_period =='seasons':
            time_cuts_1 = [[np.datetime64('%s-11-01' % year),np.datetime64('%s-02-01' % (year+1))] for year in np.arange(min_year,max_year)] # winter
            time_cuts_2 = [[np.datetime64('%s-02-01' % year),np.datetime64('%s-06-01' % (year+1))] for year in np.arange(min_year,max_year)] # spring
            time_cuts_3 = [[np.datetime64('%s-06-01' % year),np.datetime64('%s-09-01' % (year+1))] for year in np.arange(min_year,max_year)] # summer
            time_cuts_4 = [[np.datetime64('%s-09-01' % year),np.datetime64('%s-11-01' % (year+1))] for year in np.arange(min_year,max_year)] # fall
            STD = np.zeros(len(time_periods))

        STD_df = pd.DataFrame()
        for col_name in a.col_names:
            print('Calculating standard dev of hourly ramps for %s on %s basis' % (col_name,a.time_period))
            if 'Price' in col_name:
                print('OBS: taking out ramps where spot price is 0')
                if 'DK1' in col_name:
                    data_df_cut = data_df[abs(data_df['SpotPriceEUR_DK1']) > 0]
                if 'DK2' in col_name:
                    data_df_cut = data_df[abs(data_df['SpotPriceEUR_DK2']) > 0]
            if 'Con' in col_name:
                print('OBS: taking out ramps where consumption is 0')
                if 'DK1' in col_name:
                    data_df_cut = data_df[abs(data_df['GrossCon_DK1']) > 0]
                if 'DK2' in col_name:
                    data_df_cut = data_df[abs(data_df['GrossCon_DK2']) > 0]
            else:
                data_df_cut = data_df
            hourly_steps = data_df_cut['hourly_steps_'+col_name]
            datetime = data_df_cut['datetime']

            if a.time_period == 'all':
                STD = aux.get_STD(datetime,hourly_steps)
                STD_df[col_name] = [STD]
            if a.time_period == 'year':
                for i,time_cut in enumerate(time_periods):
                    STD[i] = aux.get_STD(datetime,hourly_steps,time_cut=time_cut)
                STD_df[col_name] = STD
                STD_df['time'] = [_[0].astype('datetime64[D]').item().year for _ in time_periods]
                STD_df = STD_df.set_index(STD_df['time'])        
            if a.time_period == 'seasons':
                STD_1 = np.zeros(len(time_periods)) # for winter
                STD_2 = np.zeros(len(time_periods)) # for spring
                STD_3 = np.zeros(len(time_periods)) # for fall
                STD_4 = np.zeros(len(time_periods)) # for summer
                for i,time_cut in enumerate(time_cuts_1): 
                    STD_1[i] = aux.get_STD(datetime,hourly_steps,time_cut=time_cut)
                for i,time_cut in enumerate(time_cuts_2): 
                    STD_2[i] = aux.get_STD(datetime,hourly_steps,time_cut=time_cut)
                for i,time_cut in enumerate(time_cuts_3): 
                    STD_3[i] = aux.get_STD(datetime,hourly_steps,time_cut=time_cut)
                for i,time_cut in enumerate(time_cuts_4): 
                    STD_4[i] = aux.get_STD(datetime,hourly_steps,time_cut=time_cut)
                STD_df[col_name+'_winter'] = STD_1
                STD_df[col_name+'_spring'] = STD_2
                STD_df[col_name+'_summer'] = STD_3
                STD_df[col_name+'_fall'] = STD_4

        return(STD_df)

    def GetHourlyStepPerc(self,**kwargs):
        ''' Calculates percentiles of extreme step sizes, using result from AddHourlySteps(), over specified bin size.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','time_period','time_cut','percent','scale_by_capacity','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        # Get all hourly steps
        for col_name in a.col_names:
            self.AddHourlySteps(col_name=col_name,include_storms=a.include_storms,verbose=a.verbose)

        data_df = self.data_df.copy()
        datetime = data_df.datetime.values
        if a.time_cut:
            mask = np.array([(datetime >= np.datetime64(a.time_cut[0])) & (datetime <= np.datetime64(a.time_cut[1]))])[0]
            data_df = data_df[mask]
        datetime = data_df.datetime.values

        # Identify individual years
        min_year = min(datetime).astype('datetime64[D]').item().year
        max_year = max(datetime).astype('datetime64[D]').item().year+1
        time_periods = [[np.datetime64('%s-01-01' % year),np.datetime64('%s-01-01' % (year+1))] for year in np.arange(min_year,max_year)]
        
        if a.time_period =='seasons':
            time_cuts_1 = [[np.datetime64('%s-11-01' % year),np.datetime64('%s-02-01' % (year+1))] for year in np.arange(min_year,max_year)] # winter
            time_cuts_2 = [[np.datetime64('%s-02-01' % year),np.datetime64('%s-06-01' % (year+1))] for year in np.arange(min_year,max_year)] # spring
            time_cuts_3 = [[np.datetime64('%s-06-01' % year),np.datetime64('%s-09-01' % (year+1))] for year in np.arange(min_year,max_year)] # summer
            time_cuts_4 = [[np.datetime64('%s-09-01' % year),np.datetime64('%s-11-01' % (year+1))] for year in np.arange(min_year,max_year)] # fall

        perc_dict = {}
        for name,percent in zip(['lower','upper'],[100-a.percent,a.percent]):
            perc_df = pd.DataFrame()
            perc = np.zeros(len(time_periods))
            for col_name in a.col_names:
                if a.verbose: print('Calculating %s percentiles of hourly ramps for %s on %s basis' % (percent,col_name,a.time_period))
                if a.scale_by_capacity:
                    hourly_steps = data_df['hourly_steps_'+col_name]
                else:
                    hourly_steps = data_df['hourly_raw_steps_'+col_name]
                datetime = data_df['datetime']
                if a.time_period == 'year':
                    for i,time_cut in enumerate(time_periods):
                        perc[i] = aux.get_percentile(datetime,hourly_steps,percent,time_cut=time_cut)
                    perc_df[col_name] = perc
                if a.time_period == 'seasons':
                    perc_1 = np.zeros(len(time_periods)) # for winter
                    perc_2 = np.zeros(len(time_periods)) # for spring
                    perc_3 = np.zeros(len(time_periods)) # for fall
                    perc_4 = np.zeros(len(time_periods)) # for summer
                    for i,time_cut in enumerate(time_cuts_1): 
                        # print('Now doing winter %s' % (min(time_cut).astype('datetime64[D]').item().year))
                        perc_1[i] = aux.get_percentile(datetime,hourly_steps,percent,time_cut=time_cut)
                    for i,time_cut in enumerate(time_cuts_2): 
                        # print('Now doing Spring %s' % (min(time_cut).astype('datetime64[D]').item().year))
                        perc_2[i] = aux.get_percentile(datetime,hourly_steps,percent,time_cut=time_cut)
                    for i,time_cut in enumerate(time_cuts_3): 
                        perc_3[i] = aux.get_percentile(datetime,hourly_steps,percent,time_cut=time_cut)
                    for i,time_cut in enumerate(time_cuts_4): 
                        perc_4[i] = aux.get_percentile(datetime,hourly_steps,percent,time_cut=time_cut)
                    perc_df[col_name+'_winter'] = perc_1
                    perc_df[col_name+'_spring'] = perc_2
                    perc_df[col_name+'_summer'] = perc_3
                    perc_df[col_name+'_fall'] = perc_4
            perc_df['time'] = [_[0].astype('datetime64[D]').item().year for _ in time_periods]
            perc_df = perc_df.set_index(perc_df['time'])
            perc_dict[name] = perc_df
        
        return(perc_dict)

    def GetMinimumRamps(self,col_name,time_period=False,d_data='../../data/'):
        ''' Calculates and returns all changes in power with minimum timestep in data, as fraction of total capacity in that time period %.
        '''

        # Load capacities
        capacity_ob = cap.Capacity(file_path=d_data+'ENS/',file_name='capacity_df')
        capacity_ob.AddData(file_path=d_data+'ENS/')
        if 'DK1' in col_name: capacity_df = capacity_ob.data_df_DK1 # capacity,date = capacity_ob.data_df_DK1['capacity'],capacity_ob.data_df_DK1['date']
        if 'DK2' in col_name: capacity_df = capacity_ob.data_df_DK2 # capacity,date = capacity_ob.data_df_DK2['capacity'],capacity_ob.data_df_DK2['date']
        if 'BO' in col_name: capacity_df = capacity_ob.data_df_BO # capacity,date = capacity_ob.data_df_BO['capacity'],capacity_ob.data_df_BO['date']
        if col_name == 'TotalWindPower': capacity_df = capacity_ob.data_df_DK # capacity,date = capacity_ob.data_df['capacity'],capacity_ob.data_df['date'].values
        if col_name == 'OnshoreWindPower': capacity_df = capacity_ob.data_df_onshore # capacity,date = capacity_ob.data_df['capacity'],capacity_ob.data_df['date'].values
        if col_name == 'OffshoreWindPower': capacity_df = capacity_ob.data_df_offshore # capacity,date = capacity_ob.data_df['capacity'],capacity_ob.data_df['date'].values
        # Merge capacities into data
        data_df = self.data_df[[col_name,'datetime']].copy()
        data_df_w_capacity = pd.merge(data_df,capacity_df,on='datetime',how='outer')#.set_index('datetime')
        # remove any duplicates in time
        data_df_w_capacity = data_df_w_capacity.drop_duplicates(subset='datetime')
        data_df_w_capacity = data_df_w_capacity.sort_values('datetime').reset_index(drop=True)
        # fill missing values
        data_df_w_capacity['cumsum'] = data_df_w_capacity['cumsum'].ffill() # Continue with last measured total capacity
        data_df_w_capacity[col_name] = data_df_w_capacity[col_name].ffill()
        data_df_w_capacity[col_name] = data_df_w_capacity[col_name].bfill()

        # Cut time out?
        if time_period:
            time_period_ind = [(np.datetime64(time_period[0]) <= data_df_w_capacity['datetime']) & (data_df_w_capacity['datetime'] <= np.datetime64(time_period[1]))]
            time_period_ind = np.array(time_period_ind)[0]
            data_df_w_capacity = data_df_w_capacity.loc[time_period_ind,:].reset_index()

        if len(data_df_w_capacity) > 1:
            power = data_df_w_capacity[col_name].values
            capacity = data_df_w_capacity['cumsum'].values
            datetime = data_df_w_capacity['datetime'].values
            # Calculate ramps relative to the installed capacity at the time
            ramps_MW = np.array([power[_+1] - power[_] for _ in range(len(power)-1)])
            # print('OBS: because of missing data from Bornholm, hourly changes of exactly 0 are not included')
            capacity_cut = capacity[1::]#[ramps_MW != 0]
            datetime_cut = datetime[1::]#[ramps_MW != 0]
            ramps_MW = ramps_MW#[ramps_MW != 0] 
            ramps = ramps_MW/capacity_cut*100.

        else:
            ramps_MW = [0]
            ramps = [0]
            datetime_cut = [0]
        if len(ramps) == 0:
            ramps_MW = [0]
            ramps = [0]
            datetime_cut = [0]

        ramps = np.nan_to_num(ramps)

        if not time_period:
            # Take out the first ramp value (when production drops from 0 to some value...)
            datetime_cut = datetime_cut[ramps != 0]
            ramps_MW = ramps_MW[ramps != 0]
            ramps = ramps[ramps != 0]
            ramps_MW = ramps_MW[1::]
            ramps = ramps[1::]
            datetime_cut = datetime_cut[1::]
            # Save in this object for later if they are complete
            setattr(self,'min_ramps_'+col_name+'_MW',ramps_MW)
            setattr(self,'min_ramps_'+col_name,ramps)
            setattr(self,'min_ramps_'+col_name+'_datetime',datetime_cut)

        return(ramps,datetime_cut)

    def RestoreData(self):
        ''' Restores data dataframe from a saved pandas file.
        '''
        self.data_df = pd.read_pickle(self.file_path + self.file_name + self.ext)

    def AddData(self,**kwargs):
        ''' Adds data dataframe as an attribute to the PowerData object.
        '''

        # handle default values and kwargs
        args                =   dict(file_path='',data_type='',raw_data_names='',df_cols='',col_indices='',time='',skiprows=1,append=False,merge=False,method='append')
        args                =   aux.update_dictionary(args,kwargs)
        
        self.file_path      =   args['file_path']
        self.raw_data_names =   args['raw_data_names']
        self.col_indices    =   args['col_indices']
        self.df_cols        =   args['df_cols']
        self.time           =   args['time']
        self.skiprows       =   args['skiprows']

        if hasattr(self,'data_df'):
            data_df_new = self.__ReadData()
            if args['append']: 
                self.data_df = self.data_df.append(data_df_new,ignore_index=True)
                print('appended new data to existing dataframe')
            if args['merge']:
                self.data_df = self.data_df.merge(data_df_new,on='datetime',how='outer')
                data_df = self.data_df.copy()
                # Check for duplicate columns
                for key in data_df.keys():
                    if '_x' in key:
                        data_df[key.replace('_x','')] = data_df[key].values
                        data_df = data_df.drop(key,axis=1)
                        data_df = data_df.drop(key.replace('_x','')+'_y',axis=1)
                data_df = data_df.sort_values('datetime')
                self.data_df = data_df
                print('merged new data to existing dataframe')
            self.N_datapoints = len(self.data_df)
            print('New number of datapoints: %s' % self.N_datapoints)
        else:
            data_df = self.__ReadData(**args)
            data_df = data_df.sort_values('datetime').reset_index(drop=True)
            self.data_df = data_df

        for key in self.data_df.keys():
            setattr(self,key,self.data_df[key])

    def RoundToNearestTimeStep(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','verbose','secs_or_minutes']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df             =   self.data_df.copy()
        datetimes           =   data_df.datetime
        # if a.secs_or_minutes == 'secs':
        secs                =   np.array([datetimes[_].second for _ in range(len(datetimes))])
        print('Rounding up in %s places where second counter is not 0' % (len(secs[secs != 0])))
        # change datetime to next whole minute
        secs_offset         =   60 - secs
        secs_offset[secs_offset == 60] = 0
        time_offset         =   secs_offset.astype('timedelta64[s]')
        data_df['datetime'] =   data_df['datetime'] + time_offset
        datetimes           =   data_df.datetime
        if a.secs_or_minutes == 'minutes':
            mins                =   np.array([datetimes[_].minute for _ in range(len(datetimes))])
            print('Rounding up in %s places where minute counter is not 0' % (len(mins[mins != 0])))
            # change datetime to next whole hour
            min_offset          =   60*60 - mins*60.
            min_offset[min_offset == 60*60] = 0
            time_offset         =   min_offset.astype('timedelta64[s]')
            data_df['datetime'] =   data_df['datetime'] + time_offset
        self.data_df = data_df

        print('Check number after:')
        datetimes           =   data_df.datetime
        if a.secs_or_minutes == 'secs':
            secs                =   np.array([datetimes[_].second for _ in range(len(datetimes))])
            print(len(secs[secs != 0]))
        if a.secs_or_minutes == 'minutes':
            mins                =   np.array([datetimes[_].minute for _ in range(len(datetimes))])

        # Remove duplicate times
        self.RemoveDuplicateTimes(col_name=a.col_name,verbose=a.verbose)
        
        # Remove duplicate times
        self.data_df = aux.update_times(self.data_df)

        datetimes           =   self.data_df.datetime
        mins                =   np.array([datetimes[_].minute for _ in range(len(datetimes))])

    def MatchTimeSteps(self,data_ob):
        ''' Matches time steps in self.data_df to those of the passed object "data_ob" (typically of more coarse time resolution).
        '''

        # find time steps in data_ob (e.g. 1 hour)
        data_df = data_ob.data_df.copy() 
        print('\nOriginal dataframe has %s datapoints ' % self.N_datapoints)
        new_datetime = data_df['datetime']
        new_time_step_sec = int(np.median(data_df['time_steps'].values))
        new_time_step = np.timedelta64(new_time_step_sec,'s') #s

        # interpolate power [MW] in self, by averaging over +/- half the time step:
        data_df = self.data_df.copy()
        new_data_df = pd.DataFrame()
        new_data_df['datetime'] = new_datetime
        old_datetime = data_df['datetime']
        for df_key in data_df.keys():
            if 'Power' in df_key:
                power = data_df[df_key].values
                new_power = [np.sum(power[(old_datetime > time-new_time_step) & (old_datetime < time+new_time_step)])/len(power[(old_datetime > time-new_time_step) & (old_datetime < time+new_time_step)]) for time in new_datetime] # MW
                new_data_df[df_key] = new_power
        new_data_df = aux.update_times(new_data_df)
        self.data_df = new_data_df
        self.N_datapoints = len(new_data_df)
        print('After time step matching: %s datapoints ' % self.N_datapoints)
        print('Between %s and %s' % (np.min(data_df.datetime),np.max(data_df.datetime)))

    def ChangeTimeStep(self,**kwargs):
        ''' Smoothes signal to a lower resolution time step

        Parameters
        ----------
        time_step: int
            Number of minutes to use as new time step, default: 5
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['time_step']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df = self.data_df.copy() 
        datetimes = data_df.datetime

        if np.min(data_df.time_steps.values) != np.max(data_df.time_steps.values):
            print('Min and max time step:',min(data_df.time_steps.values),max(data_df.time_steps.values))
            sys.exit("time steps are not constant, cannot proceed")
        else:
            number_to_average_over = int(np.timedelta64(str(a.time_step),'m')/np.timedelta64('%s' % (int(data_df.time_steps[0])),'s'))
            print('Going to average over %s datapoints' % number_to_average_over)

        print('data starts at: %s' % min(datetimes))

        offset = min(self.data_df.datetime).minute % a.time_step
        print('1st datapoint is offset from new time step value by %s datapoints' % offset)
        print('Total number of datapoinst: %s' % self.N_datapoints)

        indices = np.arange(self.N_datapoints/number_to_average_over) * number_to_average_over + number_to_average_over - offset
        if max(indices)+number_to_average_over > self.N_datapoints-1: 
            indices = indices[0:-1]
            print('Cutting out last time step, to not exceed length of original dataset')

        new_data_df = pd.DataFrame()
        new_data_df['datetime'] = data_df['datetime'][indices]
        for key in data_df:
            if 'time' not in key:
                array = data_df[key]
                smoothed_array = np.zeros([len(indices),number_to_average_over])
                for index in range(number_to_average_over):
                    smoothed_array[:,index] = array[indices+index]
                smoothed_array = np.sum(smoothed_array,axis=1)/number_to_average_over # take mean
                new_data_df[key] = smoothed_array
        new_data_df = aux.update_times(new_data_df)

        print('New data tail:')
        print(new_data_df.tail())
        print('Original dataframe with %s entries, condensed to %s ' % (self.N_datapoints,len(new_data_df)))

        self.data_df = new_data_df
        self.N_datapoints = len(new_data_df)

    def CutTimePeriod(self,**kwargs):

        # handle default values and kwargs
        args                =   dict(start_time=False,end_time=False,indices=False)
        args                =   aux.update_dictionary(args,kwargs)

        self.start_time     =   args['start_time']
        self.end_time       =   args['end_time']

        data_df = self.data_df.copy()

        if self.start_time:
            if self.start_time == 0: 
                time_period         =   [data_df['datetime'].values <= self.end_time]
            else: 
                time_period         =   [(self.start_time <= data_df['datetime']) & (data_df['datetime'] <= self.end_time)]
            time_period = np.array(time_period)[0]
            data_df = data_df.loc[time_period,:].reset_index()

        if args['indices']:
            data_df           =   data_df[args['indices'][0]:args['indices'][1]].reset_index()
        print('Extracted data between %s and %s' % (min(data_df['datetime']),max(data_df['datetime'])))
        print('%s datapoints' % len(data_df))

        self.data_df = data_df

    def SplitData(self,col_name,value1,value2):

        # extract original dataframe from series:
        data_df = self.data_df.copy()

        # split into two new dataframes:
        data_df_1 = data_df.loc[data_df[col_name] == value1].reset_index(drop=True)
        data_df_2 = data_df.loc[data_df[col_name] == value2].reset_index(drop=True)

        # make new dataframe with the remaining columns
        col_names = data_df.keys()
        col_names = [col_name for col_name in col_names if 'Power' not in col_name]
        new_data_df = data_df[col_names].reset_index(drop=True)
        new_data_df = new_data_df.loc[new_data_df[col_name] == value1].reset_index(drop=True)

        for key in data_df.keys():
            if ('Power' in key) | ('Price' in key) | ('GrossCon' in key):
                if key != col_name:
                    new_data_df[key+'_'+value1] = data_df_1[key].values
                    new_data_df[key+'_'+value2] = data_df_2[key].values
                    if 'Price' not in key: new_data_df[key] = data_df_1[key].values + data_df_2[key].values
                    print('Added two new columns for %s, with DK1 and DK2 partitions' % key)

        # calculate total time since beginning and time_steps
        new_data_df = aux.update_times(new_data_df).reset_index(drop=True)
        new_data_df = new_data_df.drop(col_name, 1)
        print('New dataframe has %s rows' %  len(new_data_df))

        # create new dataframe:
        setattr(self,'data_df',new_data_df)

    def AddColumnToData(self,col_name,array):

        data_df = self.data_df.copy()
        if col_name not in data_df.keys():
            data_df[col_name] = array
            self.data_df = data_df
        else:
            print('Column %s already exists in dataframe' % (col_name))

    def RemoveDuplicateTimes(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df             =   self.data_df.copy().reset_index(drop=True)    

        duplicate_times = np.unique(np.array(list(duplicates(data_df['datetime']))))

        if len(duplicate_times) == 0:
            if a.verbose: print('Found no duplicates!! :D')
        if len(duplicate_times) > 0:
            if a.verbose: print('Found duplicates at:')
            for duplicate_time in duplicate_times:
                # Identify extra rows
                if a.verbose: print(duplicate_time)
                indices_to_replace = list(np.where(data_df['datetime'] == duplicate_time)[0])
                dup_rows = data_df.loc[indices_to_replace].reset_index(drop=True)
                # Find the one that comes closest to last datapoint using col_name column
                powers = dup_rows[a.col_name]
                prev_power = data_df[a.col_name][np.min(indices_to_replace)-1]
                index_to_replace_with = (powers - prev_power).idxmin() # np.argmin
                data_df.loc[indices_to_replace[0],:] = dup_rows.loc[index_to_replace_with]
                # Remove extra rows from dataframe
                data_df = data_df.drop(data_df.index[indices_to_replace[1::]],axis=0).reset_index(drop=True)
                # print(data_df['datetime'][indices_to_replace[0::]])

        self.data_df = data_df

    def FillInTimeSteps(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   []
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df             =   self.data_df.copy().reset_index(drop=True)
        time_step           =   np.median(data_df.time_steps.values)

        index               =   np.where(data_df.time_steps.values != time_step)[0]
        if len(index) >= 1:
            data_df_temp        =   data_df.loc[min(index)-2:min(index)-1].reset_index(drop=True)
            i = 2
            for _ in index:
                N_missing_rows  =   np.ceil(data_df.time_steps[_]/time_step).astype('int64')
                for missing_row in range(N_missing_rows):
                    data_df_temp    =   data_df_temp.append(data_df.iloc[_-1],ignore_index=True)
                    new_datetime = (data_df.datetime + (1+missing_row)*np.array(time_step).astype('timedelta64[s]'))[_-1]
                    data_df_temp['datetime'][i] = new_datetime
                    i += 1
            print('Adding %s new rows of data' % len(data_df_temp))
            data_df = aux.CombineDataFrames(data_df,data_df_temp,method='merge')
        else:
            print('No missing time steps! :D')

        self.data_df = data_df

    def RemoveCrazyValues(self,col_name='',method='above',value=0):
        ''' Replaces crazy values in the dataframe with a mean of nearby points.
        '''

        data_df = self.data_df.copy()

        if method == 'above': indices_to_replace = np.where(data_df[col_name] > value)[0]
        if method == 'below': indices_to_replace = np.where(data_df[col_name] < value)[0]
        if method == 'equal': indices_to_replace = np.where(data_df[col_name] == value)[0]
        N_places = len(indices_to_replace)
        print('Replacing %s values in %s' % (N_places,col_name))

        # for index in indices_to_replace:
        for key in data_df.keys():
            if key != 'datetime':
                try:
                    data_df[key][list(indices_to_replace)] = (data_df[key][list(indices_to_replace-1)].values + data_df[key][list(indices_to_replace+1)].values)/2.
                except:
                    print('Could not replace values with mean for column %s' % key)

        self.data_df = data_df

    def ReplaceArtifacts(self,col_name,value,new_value):
        ''' Replaces weird values in the dataframe.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','value','new_value']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        try:
            N_places = len(data_df[data_df[a.col_name] == value])
            data_df[col_name][data_df[a.col_name] == value] = new_value
            print('Replaced %s in %s places with %s' % (value,N_places,new_value))
        except:
            print('data in column %s is not in str format' % a.col_name)
        self.data_df = data_df

    def FillNansNULLS(self,**kwargs):
        ''' Replaces NaNs in the dataframe with a new value.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','new_NaN_value']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        array = data_df[a.col_name].values.copy()
        N_places = len(array[array == 'NULL'])
        print('Replacing NULL in %s places with %s in %s' % (N_places,a.new_NaN_value,a.col_name))
        N_places = len(array[array == np.nan])
        print('Replacing nans in %s places with %s in %s' % (N_places,a.new_NaN_value,a.col_name))


        array[array == 'NULL'] = 0
        array = np.nan_to_num(array)

        data_df[a.col_name] = array.astype(float)
        data_df = aux.update_times(data_df)
        self.data_df = data_df

    def __ReadData(self,**kwargs):
        ''' Reads data and stores as a dataframe.
        '''

        args                =   dict(method='append')
        args                =   aux.update_dictionary(args,kwargs)
        method = args['method']

        if 'xml' in self.raw_data_names[0]:
            datafile_count = 0
            for raw_data_name in self.raw_data_names:
                tree = ET.parse( self.file_path + raw_data_name)  
                root = tree.getroot()
                print('Loading xml file at %s' % (self.file_path+raw_data_name))
                print('Number of datapoints: %s' % len(list(root)))
                # print('Column names and values per datapoint:')
                # for item in root[0]:
                #     print('%s : %s' % (item.tag,item.text))   
                data_df_temp = pd.DataFrame(columns=self.df_cols)
                for col,index in zip(self.df_cols,self.col_indices):
                    data_df_temp[col] = [node[index].text for node in root]
                # Get time stamps
                data_df_temp = data_df_temp.iloc[::-1].reset_index(drop=True)
                timestamps = [data_df_temp[self.time][_].replace('T',' ') for _ in range(len(list(root)))]
                data_df_temp['datetime'] = [dateutil.parser.parse(timestamps[i]) for i in range(len(list(root)))]
                data_df_temp = data_df_temp.drop(self.time, 1)
                # Convert all data into arrays of type float
                for key in data_df_temp.keys():
                    array = data_df_temp[key].values
                    if key != 'datetime':
                        try: 
                            data_df_temp[key] = array.astype(float)
                            print('made column %s into float' % key)
                        except:
                            print('could not make column %s into float, type: %s' % (key,type(array[0])))
                if datafile_count == 0: 
                    data_df = data_df_temp
                else:
                    data_df = aux.CombineDataFrames(data_df,data_df_temp,method=method)
                datafile_count += 1



        if 'csv' in self.raw_data_names[0]:
            datafile_count = 0
            for raw_data_name in self.raw_data_names:
                print('Loading xml file at %s' % (self.file_path+raw_data_name))
                df_cols = self.df_cols.copy()
                df_cols.append('extra')
                data_df_temp = pd.read_csv(self.file_path+raw_data_name,comment=';',sep=';;|;\*\*;',skiprows=self.skiprows,names=df_cols,engine='python')
                data_df_temp = data_df_temp.drop('extra', 1)
                # remove lines that start with ';;;;' basically:
                data_df_temp = data_df_temp.dropna(subset=['timestamp'])
                N_datapoints_temp = len(data_df_temp)
                print('Number of datapoints: %s' % N_datapoints_temp)
                timestamps = data_df_temp['timestamp1'] # because wind power timestamp is more trustworthy
                for _,timestamp in enumerate(timestamps):
                    try: 
                        PM = timestamp.find('PM')
                    except:
                        print(type(timestamp))
                        print(data_df_temp.loc[_])
                        a =aasf
                PM = [timestamp.find('PM') for timestamp in timestamps] # 0 when 'PM' is found!!
                PM = np.array(PM).astype('float')
                data_df_temp['PM'] = PM
                timestamps = [timestamp.replace(' AM','') for timestamp in timestamps]
                timestamps = [timestamp.replace(' PM','') for timestamp in timestamps]
                timestamps = [timestamp.replace('**','') for timestamp in timestamps]
                data_df_temp['datetime'] = [dateutil.parser.parse(timestamps[i]) for i in range(N_datapoints_temp)]
                for key in data_df_temp.keys():
                    if 'Power' in key:
                        power = data_df_temp[key].values
                        try:
                            power = np.array(power).astype('float')
                        except:
                            print('trying to remove weird characters in %s' % key)
                            power = [p.replace('AM','') for p in power]
                            power = [p.replace('PM','') for p in power]
                            power = [p.replace('**','') for p in power]
                            power = [p.replace(';','') for p in power]
                            data_df_temp[key] = np.array(power).astype('float')                        

                # Convert timestamps
                years = [data_df_temp['datetime'][_].year for _ in range(N_datapoints_temp)]
                months = [data_df_temp['datetime'][_].month for _ in range(N_datapoints_temp)]
                days = [data_df_temp['datetime'][_].day for _ in range(N_datapoints_temp)]
                hours = np.array([data_df_temp['datetime'][_].hour for _ in range(N_datapoints_temp)])
                mins = [data_df_temp['datetime'][_].minute for _ in range(N_datapoints_temp)]
                secs = [data_df_temp['datetime'][_].second for _ in range(N_datapoints_temp)]
                correct_hours = hours.copy()
                correct_hours = hours + 12
                correct_hours[(PM == -1)] = hours[(PM == -1)]
                correct_hours[(PM != -1) & (hours == 12)] = 12.
                correct_hours[(PM == -1) & (hours == 12)] = 0.
                data_df_temp['datetime'] = [pd.Timestamp(year=years[_], month=months[_], day=days[_], hour=correct_hours[_], minute=mins[_], second=secs[_]) for _ in range(N_datapoints_temp)]
                # Drop unnecessary columns
                data_df_temp = data_df_temp.drop('PM', 1)
                for key in data_df_temp.keys():
                    if 'timestamp' in key:
                        data_df_temp = data_df_temp.drop(key, 1)
                data_df_temp = aux.update_times(data_df_temp)
                if datafile_count == 0: 
                    data_df = data_df_temp
                else:
                    data_df = aux.CombineDataFrames(data_df,data_df_temp,method=method)
                datafile_count += 1

        self.N_datapoints = len(data_df)
        return(data_df)

class CombPowerData(PowerData):
    ''' This class defines an object that contains *combined* power info from the time series of several datasets. 
    '''

    def __init__(self,**kwargs):

        # handle default values and kwargs
        args                =   dict(ob_1=False,ob_2=False,file_path='',file_name='',ext='',dataframe=False,method=False)
        args                =   aux.update_dictionary(args,kwargs)

        if type(args['dataframe']) != bool:
            self.data_df = args['dataframe']
        else:
            self.file_path      =   args['file_path']
            self.file_name      =   args['file_name']
            self.ext            =   args['ext']

        if args['method']:
            self.data_df = aux.CombineDataFrames(args['ob_1'].data_df,args['ob_2'].data_df,**args)
        else:
            print('No method set for combining, will look for passed dataframe')
            if type(args['dataframe']) != bool:
                self.data_df = args['dataframe']
                print('Dataframe stored as attribute')
            else:
                print('No dataframe given, will look for saved file')
                try:
                    self.RestoreData()
                    print('Restored dataframe')
                except:
                    print('No stored dataframe found, could not initialize combined object')

        self.N_datapoints = len(self.data_df)



