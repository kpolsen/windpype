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
        """ Calculates accumulated capacity as function of time for DK, DK1, DK2 and Bornholm.
        """

        data_df = self.data_df.copy()

        data_df_DK1 = data_df[data_df['communenr'] > 400].copy()
        data_df_DK2 = data_df[data_df['communenr'] <= 400].copy()
        data_df_BO = data_df[data_df['communenr'] == 400].copy()
        data_df_onshore = data_df[data_df['placing'] == 'LAND'].copy()
        data_df_offshore = data_df[data_df['placing'] == 'HAV'].copy()
        data_df_DK = data_df.copy()

        data_df_DK1['accum_cap_DK1'] = np.cumsum(data_df_DK1['capacity'].values)
        data_df_DK2['accum_cap_DK2'] = np.cumsum(data_df_DK2['capacity'].values)
        data_df_BO['accum_cap_BO'] = np.cumsum(data_df_BO['capacity'].values)
        data_df_onshore['accum_cap_onshore'] = np.cumsum(data_df_onshore['capacity'].values)
        data_df_offshore['accum_cap_offshore'] = np.cumsum(data_df_offshore['capacity'].values)
        data_df_DK['accum_cap_DK'] = np.cumsum(data_df['capacity'].values)

        cap_data_df = data_df_DK
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = pd.merge(cap_data_df,data_df_DK1,on='datetime',how='outer')
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = pd.merge(cap_data_df,data_df_DK2,on='datetime',how='outer')
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = pd.merge(cap_data_df,data_df_BO,on='datetime',how='outer')
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = pd.merge(cap_data_df,data_df_onshore,on='datetime',how='outer')
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = pd.merge(cap_data_df,data_df_offshore,on='datetime',how='outer')
        cap_data_df = cap_data_df.drop(['commune','communenr','placing','capacity'],axis=1)
        cap_data_df = cap_data_df.sort_values('datetime').reset_index(drop=True)
        cap_data_df = cap_data_df.fillna(method='ffill')
        cap_data_df = cap_data_df.fillna(-1)

        self.cap_data_df = cap_data_df

    def PlotCapacity(self,**kwargs):
        """ Plots accumulated capacity as function of time for DK, DK1, DK2 and Bornholm.
        """

        # create a custom namespace for this method
        argkeys_needed      =   ['fig_name','time_cut','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        if not hasattr(self, 'cap_data_df'): self.AccumulateCapacity()

        cap_data_df = self.cap_data_df.copy()
        print('Total capacity in DK: %s MW' % np.max(cap_data_df['accum_cap_DK']))

        mask = np.array([(cap_data_df.datetime > a.time_cut[0]) & (cap_data_df.datetime < a.time_cut[1])])[0]
        print('len of capacity arrays: %s' % (len(mask)))

        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel('Time',fontsize=20)
        ax1.set_ylabel('Accumulative capacity [MW]',fontsize=20)
        colors = ['k','b','r','purple']
        for _,ext in enumerate(['DK','DK1','DK2','BO']):
            ax1.plot(cap_data_df['datetime'],cap_data_df['accum_cap_'+ext],colors[_],label=ext)
            ax1.plot([max(cap_data_df['datetime']),a.time_cut[1]],2*[max(cap_data_df['accum_cap_'+ext])],colors[_]) # Filling up until now...
            print('%s takes up:' % ext)
            print('%.2f %% to %.2f %% ' % (min(cap_data_df[mask]['accum_cap_'+ext]/cap_data_df[mask]['accum_cap_DK'])*100,max(cap_data_df[mask]['accum_cap_'+ext]/cap_data_df[mask]['accum_cap_DK'])*100))  
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

        capacity = pd.read_table(self.file_path+'Kapacitet.txt',skiprows=1,names=['capacity'],engine='python',skip_blank_lines=False)
        placing = pd.read_table(self.file_path+'Placering.txt',skiprows=1,names=['placing'],engine='python',skip_blank_lines=False)
        communenr = pd.read_table(self.file_path+'Kommune-nr.txt',skiprows=1,names=['communenr'],engine='python',skip_blank_lines=False)
        commune = pd.read_table(self.file_path+'Kommune.txt',skiprows=1,names=['commune'],engine='python',skip_blank_lines=False)
        date = pd.read_table(self.file_path+'Dato.txt',skiprows=1,names=['datetime'],engine='python',skip_blank_lines=False)
        capacity['capacity'] = capacity['capacity']/1000. # MW

        # Convert dates
        self.N_datapoints = len(date)
        datetime = [dateutil.parser.parse(date['datetime'][i]) for i in range(self.N_datapoints)]

        # Convert placing
        placing['placing'] = placing['placing'].str.upper()

        # Combine into one dataframe
        data_df = pd.concat([capacity,communenr,commune,placing],axis=1)
        data_df['datetime'] = datetime
        data_df = data_df.sort_values('datetime').reset_index(drop=True)

        if time_period:
            time_period = [(time_period[0] <= data_df['datetime']) & (data_df['datetime'] <= time_period[1])]
            time_period = np.array(time_period)[0]
            data_df = data_df.loc[time_period,:].reset_index()

        return(data_df)