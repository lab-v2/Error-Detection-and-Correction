import numpy as np
import pandas as pd

from tqdm import tqdm
from geopy import distance as gpd

class DataPrep:
    def __init__(self, data_path, save_path):
        self.df = pd.read_csv(data_path)
        self.save_path = save_path

    def pre_process(self, transport_modes):
        self.df = self.df[self.df['Label'].isin(transport_modes)]

        # Get Date information
        self.df['Date_Time'] = pd.to_datetime(self.df['Date_Time'])
        self.df['TimeDelta_secs'] = self.df['Date_Time'].diff().dt.seconds
        self.df['TimeDelta_secs'] = self.df['TimeDelta_secs'].fillna(0)
        self.df['TimeDelta_mins'] = self.df['TimeDelta_secs']//60
        return self.df

    def create_segments(self, dataframe, thresh):
        # Create Segment
        segment=1
        segment_list = []
        for userid in tqdm(dataframe['Id_user'].unique(), desc='Creating segments'):
            user_df = dataframe[dataframe['Id_user'] == userid].reset_index(drop = True)
            curr_mode = user_df.loc[0, 'Label']
            for rec in user_df.iterrows():
                if (rec[1]['Label'] == curr_mode) and (rec[1]['TimeDelta_mins']<thresh):
                    segment_list.append(segment)
                elif (rec[1]['Label'] == curr_mode) and (rec[1]['TimeDelta_mins']>thresh):
                    segment+=1
                    segment_list.append(segment)
                else:
                    curr_mode = rec[1]['Label']
                    segment+=1
                    segment_list.append(segment)

        # Segment the GPS traces
        dataframe['segment'] = segment_list
        self.df = dataframe
        return self.df

    def compute_distance(self, dataframe, metric = 'm'):
        # Compute distance from two coordinate points
        interval_dist = [0]
        tempdf = dataframe.reset_index(drop = True)
        for idx in range(1, len(dataframe)):
            lon1 = tempdf.loc[idx-1]['Longitude']
            lat1 = tempdf.loc[idx-1]['Latitude']
            lon2 = tempdf.loc[idx]['Longitude']
            lat2 = tempdf.loc[idx]['Latitude']
            coords_1 = (lat1, lon1)
            coords_2 = (lat2, lon2)
            if metric == 'km':
                dist = gpd.geodesic(coords_1, coords_2).km
            elif metric == 'm':
                dist = gpd.geodesic(coords_1, coords_2).km*1000
            else:
                dist = gpd.geodesic(coords_1, coords_2).miles
            interval_dist.append(dist)
        return interval_dist
    
    def compute_finite_difference(self, dataframe, interval_column, time_column, diff = False):
        # Compute Finite Differences to get GPS features
        tempdf = dataframe.reset_index(drop = True)
        finite_diff_value_list = [0] 
        for idx in range(1, len(dataframe)):
            val_i = tempdf.loc[idx][interval_column]
            time_diff = tempdf.loc[idx][time_column]
            if diff:
                val_iminus1 = tempdf.loc[idx-1][interval_column]
                if time_diff == 0:
                    finite_difference = np.nan
                else:
                    finite_difference = (val_i - val_iminus1)/time_diff
            else:
                if time_diff == 0:
                    finite_difference = np.nan
                else:
                    finite_difference = val_i/time_diff
            finite_diff_value_list.append(finite_difference)
        return finite_diff_value_list
    

    def get_statistical_features(self, dataframe, column):
        # Get Statistical Features
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        median_val = dataframe[column].median()
        mad_value = (dataframe[column] - dataframe[column].mean()).abs().mean()
        iqr_value = dataframe[column].quantile(0.75) - dataframe[column].quantile(0.25)
        pc75_val = dataframe[column].quantile(0.75)
        pc90_val = dataframe[column].quantile(0.9)
        pc95_val = dataframe[column].quantile(0.95)
        dataframe['bins'] = pd.cut(dataframe[column], 6, include_lowest = True, right = True, 
                                labels = [1, 2, 3, 4, 5, 6], duplicates='drop')
        hist1_val = dataframe['bins'].isin([1]).sum()/len(dataframe)
        hist2_val = dataframe['bins'].isin([1, 2]).sum()/len(dataframe)
        hist3_val = dataframe['bins'].isin([1, 2, 3]).sum()/len(dataframe)
        hist4_val = dataframe['bins'].isin([1, 2, 3, 4]).sum()/len(dataframe)
        hist5_val = dataframe['bins'].isin([1, 2, 3, 4, 5]).sum()/len(dataframe)
        return [mean_val, std_val, median_val, mad_value, iqr_value, pc75_val, pc90_val, pc95_val, 
                hist1_val, hist2_val, hist3_val, hist4_val, hist5_val]
    
    def extract_features(self, dataframe):
        # EXTRACT FEATURES
        feature_records_for_segments = []
        feature_cols = []
        gps_feature_list = ['interval_vel_ms', 'interval_acc_ms2', 'pos_int_acc_set', 'int_dec_set', 'pos_vel_set']

        for gps_feature in gps_feature_list:
            for stat_feature in ['mean', 'std', 'median', 'mad', 'iqr', '75pc', '90pc', '95pc', 
                                'hist1', 'hist2', 'hist3', 'hist4', 'hist5']:
                feature_cols.append(gps_feature + '_' + stat_feature)
                
        feature_cols = ['segment'] + feature_cols + ['label']
                

        for segment in tqdm(dataframe['segment'].unique()[:], desc='Extract features'):
            
            # Iterate through each segment and filter them
            segment_df = dataframe[dataframe['segment'] == segment].reset_index(drop=True).copy()
            label = segment_df['Label'].iloc[0]
            
            # ------GET GPS FEATURES MENTIONED IN gps_feature_list------
            # Compute Distance
            segment_df['distance'] = self.compute_distance(segment_df)
            # Compute Interval Velocity
            segment_df['interval_vel_ms'] = self.compute_finite_difference(segment_df, 
                                                                interval_column = 'distance', 
                                                                time_column = 'TimeDelta_secs')
            # Compute Interval Acceleration
            segment_df['interval_acc_ms2'] = self.compute_finite_difference(segment_df, 
                                                                interval_column = 'interval_vel_ms', 
                                                                time_column = 'TimeDelta_secs', 
                                                                diff = True)
            # Compute Positive Interval Acceleration
            segment_df['pos_int_acc_set'] = segment_df['interval_acc_ms2'].apply(lambda x: x if x>0 else np.nan)
            
            # Compute Interval Deceleration
            segment_df['int_dec_set'] = segment_df['interval_acc_ms2'].apply(lambda x: np.nan if x>0 else x)
            
            # Compute Positive Interval Speed
            segment_df['pos_vel_set'] = segment_df['interval_vel_ms'].apply(lambda x: x if x>0 else np.nan)

            # Stat features for distance
            stat_feature_records = [segment]
            for gps_feature in gps_feature_list:
                try:
                    stat_feature_records+=self.get_statistical_features(segment_df, gps_feature)
                except:
                    pass
            feature_records_for_segments.append(stat_feature_records + [label])

        # Final Output
        self.df = pd.DataFrame(feature_records_for_segments, columns = feature_cols)
        self.df = self.df.dropna()
        self.df.to_csv(self.save_path, index=False)
        self.df = self.df.dropna() 
        return self.df