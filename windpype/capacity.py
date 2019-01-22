###
### Submodule capacity
###

print('windpype submodule "capacity" imported')

import windpype.aux as aux
import numpy as np
import pandas as pd
import dateutil.parser
import xml.etree.ElementTree as ET 
from scipy import signal
import scipy.integrate as integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as mtransforms
import seaborn as sns
import datetime as dt

d_plot = 'plots/'

class Capacity():
    """ This class defines an object that contains the information on installed capacity as found on ENS.dk
    """

    def __init__(self,**kwargs):

        # handle default values and kwargs
        args                =   dict(file_path='',file_name='')
        args                =   aux.update_dictionary(args,kwargs)
        self.file_path      =   args['file_path']
        self.file_name      =   args['file_name']

    def AddData(self,**kwargs):
        """ Adds data dataframe as an attribute to the PowerData object.
        """

        # handle default values and kwargs
        args                =   dict(file_path='')
        args                =   aux.update_dictionary(args,kwargs)
        
        self.file_path      =   args['file_path']

        try:
            data_df = pd.read_pickle(self.file_path + self.file_name)
            # print('Data has been read in before, reloading dataframe at: ')
            # print(self.file_path + self.file_name)
        except:
            print('No series found, going to read data...')
            data_df = self.__ReadData()
            data_df.to_pickle(self.file_path + self.file_name)

        self.data_df = data_df
        self.AccumulateCapacity()

    def AccumulateCapacity(self):
        """ Calculates accumulated capacity as function of time for all regions.
        """

        data_df = self.data_df.copy()
        cap_data_df = data_df[['datetime','datetime_end','capacity']].copy()

        # years with production info:
        years = list(range(1977,2019))

        # For all of DK
        end_df = pd.DataFrame({'datetime':cap_data_df['datetime_end'],'capacity':-1.* cap_data_df['capacity'].values,'datetime_end':cap_data_df['datetime_end']})
        cap_data_df = cap_data_df.append(end_df,sort=True)
        cap_data_df = cap_data_df.groupby(by=cap_data_df['datetime']).sum().reset_index()
        cap_data_df['accum_cap_DK'] = np.cumsum(cap_data_df['capacity'].values)

        setattr(self,'accum_cap_DK',np.cumsum(cap_data_df['capacity'].values))
        setattr(self,'accum_cap_DK_datetime',cap_data_df['datetime'])

        for region in ['DK1','DK2','BO','onshore','offshore']:
            data_df_copy = data_df.copy()
            if region == 'DK1': data_df_copy = data_df_copy[data_df_copy['communenr'] > 400]
            if region == 'DK2': data_df_copy = data_df_copy[data_df_copy['communenr'] <= 400]
            if region == 'BO': data_df_copy = data_df_copy[data_df_copy['communenr'] == 400]
            if region == 'onshore': data_df_copy = data_df_copy [data_df['placing'] == 'LAND'].copy()
            if region == 'offshore': data_df_copy = data_df_copy [data_df['placing'] == 'HAV'].copy()
            data_df_copy = data_df_copy[['datetime','capacity','datetime_end']]
            # Add end of turbine production as negative capacity
            end_df = pd.DataFrame({'datetime':data_df_copy['datetime_end'],'capacity':-1.* data_df_copy['capacity'].values,'datetime_end':data_df_copy['datetime_end']})
            data_df_copy.append(end_df,sort=True)
            data_df_copy = data_df_copy.groupby(by=data_df_copy['datetime']).sum().reset_index()
            # data_df_copy = data_df_copy.sort_values('datetime').reset_index(drop=True)
            setattr(self,'accum_cap_%s' % region,np.cumsum(data_df_copy['capacity'].values))
            setattr(self,'accum_cap_%s_datetime' % region,data_df_copy['datetime'])
            data_df_copy['accum_cap_%s' % region] = np.cumsum(data_df_copy['capacity'].values)
            setattr(self,'data_cap_df_%s' % region,data_df_copy)

        self.cap_data_df = cap_data_df

    def PlotCapacity(self,**kwargs):
        """ Plots accumulated capacity as function of time for DK, DK1, DK2 and Bornholm.
        """

        # create a custom namespace for this method
        argkeys_needed      =   ['fig_name','time_cut','fig_format','regions']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        if not hasattr(self, 'cap_data_df'): self.AccumulateCapacity()

        # cap_data_df = self.cap_data_df.copy()
        print('Total capacity in DK: %s MW' % np.max(self.accum_cap_DK))

        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel('Time',fontsize=20)
        ax1.set_ylabel('Accumulative capacity [MW]',fontsize=20)
        colors = ['k','b','r','purple']
        for _,region in enumerate(a.regions):
            datetime = getattr(self,'accum_cap_%s_datetime' % region)
            accum_cap = getattr(self,'accum_cap_%s' % region)
            ax1.plot(datetime,accum_cap,colors[_],label=region)
            ax1.plot([max(datetime),a.time_cut[1]],2*[max(accum_cap)],colors[_]) # Filling up until now...
            # print('%s takes up:' % ext)
            # print('%.2f %% to %.2f %% ' % (min(accum_cap/self.accum_cap_DK)*100,max(accum_cap/self.accum_cap_DK)*100))  
        if a.time_cut: ax1.set_xlim(a.time_cut)
        plt.legend(fontsize=15)
        ax1.grid()

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=500)

    def PlotCapacityOrigin(self,**kwargs):
        """ Plots origin (kommune) of installed capacity for a certain time period, sorted by decreasing capacity.
        """

        # create a custom namespace for this method
        argkeys_needed      =   ['fig_name','time_cut','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        fig = plt.figure(figsize=(12,10))
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylabel('Location')
        ax1.set_xlabel('Accumulative Capacity [MW]')

        data_df_cut = self.__ReadData(time_period=a.time_period)

        communes = np.unique(data_df_cut['commune'].values)
        capacity = np.zeros(len(communes))
        for i,commune in enumerate(communes):
            capacity[i] = np.sum(data_df_cut[data_df_cut['commune'] == commune]['capacity'])

        capacity.sort()
        offsets = np.arange(i+1)+1
        [ax1.barh(y,x,0.5) for y,x in zip(offsets,capacity)]

        ax1.set_xscale('log')
        ax1.set_yticks(np.arange(i+1)+1)
        ax1.set_yticklabels(np.unique(data_df_cut['commune'].values))

        if a.fig_name: 
            plt.savefig(d_plot+'capacity_origin'+'.'+a.fig_format, format=a.fig_format,dpi=300)

    def __ReadData(self,time_period=False):
        """ Loads data and stores as a dataframe.
        """

        # years with production info:
        years = list(range(1977,2019))

        capacity_df = pd.read_excel(self.file_path+'anlaegprodtilnettet.xls',skiprows=17,skipfooter=3)
        # rewrite column names, starting with production years (and months of most recent year)
        production_dates = pd.read_excel(self.file_path+'anlaegprodtilnettet.xls',skiprows=15,nrows=1)
        capacity_keys = list(capacity_df.keys())[0:16]
        production_dates = production_dates.iloc[0].values
        for _,date in enumerate(production_dates):
            try:
                production_dates[_] = date.month
            except:
                pass
        production_dates = list(production_dates)[16::]
        capacity_keys.extend(production_dates)
        translate = {'Møllenummer (GSRN)' : 'GSRN',\
                        'Dato for oprindelig nettilslutning' : 'datetime',\
                        'Kapacitet (kW)' : 'capacity',\
                        'Kommune-nr.' : 'communenr',\
                        'Kommune' : 'commune',\
                        'Type af placering' : 'placing',\
                        'Akkumuleret' : 'prod2018'}
        for year in years:
            if year != 2018: translate[year] = 'prod%s' % year
        for _ in range(len(capacity_keys)):
            if capacity_keys[_] in translate.keys():
                capacity_keys[_] = translate[capacity_keys[_]]
        capacity_df.columns = capacity_keys

        # Select only certain columns
        data_df = capacity_df[['GSRN','datetime','capacity','communenr','commune','placing']].copy()
        for year in years:
            data_df['prod%s' % year] = capacity_df['prod%s' % year]

        # Convert datetimes and sort
        self.N_datapoints = len(data_df)
        # datetime = [dateutil.parser.parse(str(capacity_df['datetime'][i])) for i in range(self.N_datapoints)]
        # capacity_df['datetime'] = datetime
        data_df = data_df.sort_values('datetime').reset_index(drop=True)

        # Convert placing
        data_df['placing'] = data_df['placing'].str.upper()

        # Set no production to 0, remove NaNs
        for year in years:
            data_df['prod%s' % year] = data_df['prod%s' % year].fillna(0)

        # Add end date of working life
        datetime_end = np.array([np.datetime64('2019-01-01 00:00:00')]*len(data_df))
        # years.reverse()
        total_prod = 0
        for year in years:
            prod = data_df['prod%s' % year].values
            datetime_end[prod > 0] = np.datetime64('%s-01-01 00:00:00' % (year+1)) # production ran until the end of this year
            total_prod += prod
        data_df['datetime_end'] = datetime_end
        # set capacity to 0 for turbines with no reported production
        capacity = data_df['capacity'].values
        capacity[total_prod == 0] = 0
        data_df['capacity'] = capacity

        # Forward fill any other NaNs:
        data_df = data_df.fillna(method='ffill')
        data_df = data_df.fillna(-1)

        if time_period:
            time_period = [(time_period[0] <= data_df['datetime']) & (data_df['datetime'] <= time_period[1])]
            time_period = np.array(time_period)[0]
            data_df = data_df.loc[time_period,:].reset_index()

        return(data_df)