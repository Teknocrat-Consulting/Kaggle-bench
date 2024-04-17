import logging
import pandas as pd
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from feature_engine.creation import CyclicalFeatures
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from feature_engine.encoding import CountFrequencyEncoder
import os



class Data_Preprocess():
    def __init__(self, df, name):
        self.name = name
        self.df = df
        #self.df = self.df.sample(5000)
        self.date_formats = (
        '%b-%y','%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d', '%B %d, %Y', '%d-%b-%Y','%b %d, %Y', '%Y-%m',         
        '%m/%Y','%B %Y', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M:%S','%m/%d/%Y %I:%M %p','%Y-%m-%dT%H:%M:%S''%a, %d %b %Y %H:%M:%S GMT', '%A, %B %d, %Y'   
        )

       
        self.logger = self.setup_logger()
        self.log_file = os.path.join(os.getcwd(), self.name, f"{self.name}_logfile.log")
        # Ensure the directory exists
        os.makedirs(os.path.join(os.getcwd(), self.name), exist_ok=True)

        
        # Configure logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info("Original shape: {}".format(self.df.shape))
        #logging.info("Data information: {}".format(self.df.info()))

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(f'{self.name}_logfile.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def drop_columns_with_high_nan(self, df, threshold=70):
        try:
            df_ = df.copy()
            nan_percentages = df.isnull().mean() * 100
            columns_to_drop = nan_percentages[nan_percentages > threshold].index
            if columns_to_drop.any():
                self.logger.info("Dropped columns with high null values: {}".format(columns_to_drop))
                return df.drop(columns=columns_to_drop)
            else:
                return df_
        except Exception as e:
            self.logger.info(f"Error in drop_columns_with_high_nan : {e}")
        
    def parse_dates(self,date_str):
        for fmt in self.date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                return pd.NaT 
            
    def clean_numeric_value(self,value):
        # Remove commas and any non-numeric characters except dot, percent, and minus sign
        value=str(value)
        cleaned_value = re.sub(r'[^\d.]', '', value)
        return str(cleaned_value)


    def change_datatype(self, df, column_name, new_datatype):
        try:
            if new_datatype == 'int':
                df[column_name] =  df[column_name].apply(self.clean_numeric_value)
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(int)
            elif new_datatype == 'float':
                df[column_name] =  df[column_name].apply(self.clean_numeric_value)
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(float)
            elif new_datatype == 'str':
                df[column_name] = df[column_name].astype(str)
            elif new_datatype == 'bool':
                df[column_name] = df[column_name].astype(bool)
            elif new_datatype == 'datetime':
                df[column_name] =  df[column_name].apply(self.parse_dates)
            else:
                logging.warning("Unsupported data type.")
        except Exception as e:
            self.logger.info("Error converting column '{}' to {}: {}".format(column_name, new_datatype, e))

    def is_time_series(self,df, time_column=None, threshold=0.9):
        # Check if the dataset has a time column
        if time_column is not None:
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found in the dataset.")
            else:
                # Ensure time column is in datetime format
                df[time_column] = pd.to_datetime(df[time_column])
                # Check if data is ordered chronologically
                is_ordered = df[time_column].is_monotonic_increasing
                print(is_ordered)
                # Check if data covers at least 90% of the time span (for stationarity)
                time_span = df[time_column].max() - df[time_column].min()
                coverage_ratio = df[time_column].nunique() / ((time_span.days + 1) if time_span.days > 0 else 1)
                is_stationary = coverage_ratio >= threshold
                print(is_stationary)
                return is_ordered and not is_stationary
        
        return False

    def split_data(self, df, target_column, test_size=0.2,threshold = 30):
        df.dropna(subset=[target_column],inplace=True)
        X = df.drop(columns=[target_column]) 
        y = df[target_column] 

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
            xy_test = pd.concat([X_test,y_test],axis=1)
            xy_test.dropna(inplace=True)
            X_test = xy_test.drop(columns=[target_column]) 
            y_test = xy_test[target_column]
            self.logger.info("This is a Classification problem")
            return X_train, X_test, y_train, y_test,False
    
        except ValueError as e:
            if "The least populated class in y has only 1 member" in str(e):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                xy_test = pd.concat([X_test,y_test],axis=1)
                xy_test.dropna(inplace=True)
                X_test = xy_test.drop(columns=[target_column]) 
                y_test = xy_test[target_column]
                self.logger.info("This is a Regression problem")
                return X_train, X_test, y_train, y_test,True
            else:
                self.logger.info("Another ValueError occurred in split_data")

    def low_variance_features(self, X_train, X_test, variables=None, threshold=0.80):
        try:
            x_train = X_train.copy() 
            x_test = X_test.copy()
            earlier_cols = list(X_train.columns)
            dcf = DropConstantFeatures(tol=threshold, missing_values='ignore', variables=variables)
            X_train_ = dcf.fit_transform(X_train)
            later_cols = list(X_train_.columns)
            diff = list(set(earlier_cols) - set(later_cols))
            if len(diff) > 0:
                X_test_ = X_test.drop(diff, axis=1)
                self.logger.info(f"Features with low variance dropped [{diff}] with Threshold = 0.8")
                return X_train_, X_test_
            else:
                return x_train,x_test
        except Exception as e:
            self.logger.info(f"Error in low_variance_features : {e}")
            return x_train,x_test

    def high_correlation_features(self, X_train, X_test, variables=None, threshold=0.9):
        try:
            x_train = X_train.copy() 
            x_test = X_test.copy()
            earlier_cols = list(X_train.columns)
            if len(self.column_datatypes(X_train,earlier_cols)['numerical'])>=2:
                dcf = DropCorrelatedFeatures(threshold=threshold, missing_values='ignore', variables=variables)
                X_train_ = dcf.fit_transform(X_train)
                later_cols = list(X_train_.columns)
                diff = list(set(earlier_cols) - set(later_cols))
                if len(diff) > 0 :
                    X_test_ = X_test.drop(diff, axis=1)
                    self.logger.info(f"Features with high correlation dropped [{diff}] with Threshold = 0.9")
                    return X_train_, X_test_
                else:
                    return x_train,x_test
            else:
                    return x_train,x_test
        except Exception as e:
            self.logger.info(f"Error in high_correlation_features : {e}")
            return x_train,x_test

    def column_datatypes(self, df, columns):
        result = {'object': [], 'numerical': []}
        for column in columns:
            dtype = df[column].dtype
            if dtype == 'object':
                result['object'].append(column)
            elif dtype == 'int64' or dtype == 'float64':
                result['numerical'].append(column)
        return result

    def numerical_missing_imputation(self, df, only_numerical, method="median"):
        try:
            if len(only_numerical) > 0:
                mmi = MeanMedianImputer(imputation_method=method, variables=only_numerical)
                mmi.fit(df[only_numerical])
                df[only_numerical] = mmi.transform(df[only_numerical])
                self.logger.info("Numerical Null values imputation done with Median")
                return df
        except Exception as e:
            self.logger.info(f"Error in numerical_missing_imputation : {e}")
            return df

    def categorical_missing_imputation(self, df, dict_, count=10, fill_value=" "):
        try:
            freq_cols = [x for x in dict_ if dict_[x] <= count]
            empty_string_cols = [x for x in dict_ if dict_[x] > count]
            #print("freq_cols : ",freq_cols)
            #print("empty_string_cols : ", empty_string_cols)

            if len(freq_cols) > 0 and len(empty_string_cols) > 0:

                ci_freq = CategoricalImputer(imputation_method='frequent', variables=freq_cols)
                ci_freq.fit(df[freq_cols])
                df[freq_cols] = ci_freq.transform(df[freq_cols])

                ci_empty_string = CategoricalImputer(imputation_method='missing', fill_value=fill_value,
                                                    variables=empty_string_cols)
                ci_empty_string.fit(df[empty_string_cols])
                df[empty_string_cols] = ci_empty_string.transform(df[empty_string_cols])
                self.logger.info("Categorical Null values imputation done")
                return df

            elif len(freq_cols) > 0 and len(empty_string_cols) == 0:
                ci_freq = CategoricalImputer(imputation_method='frequent', variables=freq_cols)
                ci_freq.fit(df[freq_cols])
                df[freq_cols] = ci_freq.transform(df[freq_cols])
                self.logger.info("Categorical Null values imputation done")
                return df

            elif len(empty_string_cols) > 0 and len(freq_cols) == 0:
                ci_freq = CategoricalImputer(imputation_method='frequent', variables=freq_cols)
                ci_freq.fit(df[freq_cols])
                df[freq_cols] = ci_freq.transform(df[freq_cols])
                self.logger.info("Categorical Null values imputation done")
                return df
        except Exception as e:
            self.logger.info(f"Error in categorical_missing_imputation : {e}")
            return df

    def outlier_imputation(self, df, outliers_list):
        try:
            wz = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=outliers_list)
            wz.fit(df[outliers_list])
            df[outliers_list] = wz.transform(df[outliers_list])
            return df
        except Exception as e:
            self.logger.info(f"Error in outlier_imputation : {e}")
            return df

    def category_encode_dict(self, X_train):
        result = {'object': [], 'numerical': [], 'datetime': []}
        for column in list(X_train.columns):
            dtype = X_train[column].dtype
            if dtype == 'object':
                result['object'].append(column)
            elif dtype == 'int64' or dtype == 'float64':
                result['numerical'].append(column)
            elif dtype == "datetime64[ns]":
                result['datetime'].append(column)
        return result

    def categorical_encoding(self, X_train, X_test, dict_, unique_count=30):
        try:
            OHE_cols = [x for x in dict_ if dict_[x] <= unique_count]
            frequency_count_cols = [x for x in dict_ if dict_[x] > unique_count]

            # print("OHE_cols : ",OHE_cols)
            # print("OHE_cols length: ",len(OHE_cols))
            # print("frequency_count_cols : ",frequency_count_cols)
            # print("frequency_count_cols length: ",len(frequency_count_cols))

            if len(OHE_cols)>0 and len(frequency_count_cols)>0:
                #print("1 called")
                encoder_ohe = OneHotEncoder(variables=OHE_cols, ignore_format=True)
                X_train = encoder_ohe.fit_transform(X_train)
                X_test = encoder_ohe.transform(X_test)

                encoder_freq = CountFrequencyEncoder(encoding_method='frequency', variables=frequency_count_cols,
                                                    ignore_format=True)
                X_train = encoder_freq.fit_transform(X_train)
                X_test = encoder_freq.transform(X_test)
                self.logger.info("Categorical Encoding done")
                return X_train,X_test

            elif len(frequency_count_cols) > 0 and len(OHE_cols) == 0:
                #print("2 called")
                encoder_freq = CountFrequencyEncoder(encoding_method='frequency', variables=frequency_count_cols,
                                                    ignore_format=True)
                X_train = encoder_freq.fit_transform(X_train)
                X_test = encoder_freq.transform(X_test)
                self.logger.info("Categorical Encoding done")
                return X_train,X_test

            elif len(OHE_cols) > 0 and len(frequency_count_cols) == 0:
            # print("3 called")
                encoder_ohe = OneHotEncoder(variables=OHE_cols, ignore_format=True)
                X_train = encoder_ohe.fit_transform(X_train)
                X_test = encoder_ohe.transform(X_test)
                self.logger.info("Categorical Encoding done")
                return X_train,X_test
        except Exception as e:
            self.logger.info(f"Error in categorical_encoding : {e}")
            return X_train,X_test    
        

    def cyclical_encoding(self, Xy_train, Xy_test, date_col, target_column):
        try:
            Xy_train["Month"] = pd.DatetimeIndex(Xy_train[date_col]).month
            Xy_train["Day"] = pd.DatetimeIndex(Xy_train[date_col]).day

            Xy_test["Month"] = pd.DatetimeIndex(Xy_test[date_col]).month
            Xy_test["Day"] = pd.DatetimeIndex(Xy_test[date_col]).day

            Xy_train.dropna(subset=[date_col], inplace=True)
            Xy_test.dropna(subset=[date_col], inplace=True)

            X_train = Xy_train.drop(columns=[target_column])
            y_train = Xy_train[target_column]

            X_test = Xy_test.drop(columns=[target_column])
            y_test = Xy_test[target_column]

            cyclical = CyclicalFeatures(variables=["Month", "Day"], drop_original=True)
            X_train = cyclical.fit_transform(X_train)
            X_test = cyclical.transform(X_test)

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            self.logger.info(f"Error in cyclical_encoding : {e}")
            return None

    def scale_features(self, X_train, X_test, features, method='normalization'):
        try:
            if method == 'normalization':
                scaler = MinMaxScaler()
            elif method == 'standardization':
                scaler = StandardScaler()
            else:
                raise ValueError("Unsupported scaling method. Please use 'normalization' or 'standardization'.")

            X_train[features] = scaler.fit_transform(X_train[features])
            X_test[features] = scaler.transform(X_test[features])
            return X_train, X_test
        except Exception as e:
            self.logger.info(f"Error in scale_features : {e}")
            return X_train, X_test

    def encode_target_variable(self, y_train, y_test):
        try:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            return pd.DataFrame(y_train_encoded), pd.DataFrame(y_test_encoded)
        except Exception as e:
            self.logger.info(f"Error in encode_target_variable : {e}")
            return pd.DataFrame(y_train), pd.DataFrame(y_test)

    def run(self, change_datatype=False, column_name=None, new_datatype=None, target_column=None):
        # DROP features with high Nan
        nan_dropped = self.drop_columns_with_high_nan(self.df)

        # Create a folder to store preprocessing results
        folder_name = self.name.split('.')[0]  # Extracting name without extension
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Change current working directory to the created folder
        os.chdir(folder_name)

        # Change Datatype
        if change_datatype and column_name and new_datatype:
            column_name = [i.strip() for i in column_name.split(',')]
            new_datatype = [i.strip() for i in new_datatype.split(',')]
            for cols, new in zip(column_name, new_datatype):
                self.change_datatype(nan_dropped, cols, new)

            self.logger.info("New data information after changing datatypes: {}".format(nan_dropped.info()))

         ## Check if timeseries data or not:
    
        result_time = self.category_encode_dict(nan_dropped)
        if len(result_time['datetime']) > 0:
            date_to_use = ""
            if len(result_time['datetime']) > 1:
                dict_count = dict(X_train[result_time['datetime']].isnull().sum())
                min_key = min(dict_count, key=lambda k: dict_count[k])
                date_to_use += min_key
                date_to_use.strip()

            else:
                date_to_use += result_time['datetime'][0]
                date_to_use.strip()
            print("date used : ",date_to_use)

            if self.is_time_series(nan_dropped,date_to_use) == True and len(date_to_use)>0:
                print("The dataset is timseries data. Preprocessing not available right now.")
                self.logger.info("The dataset is timseries data. Preprocessing not available right now.")

        else:
        # Split the Data
            if target_column:
                target_column = target_column.strip()
                X_train, X_test, y_train, y_test,is_regression = self.split_data(nan_dropped, target_column, test_size=0.2)
                self.logger.info("Training set (X): {}".format(X_train.shape))
                self.logger.info("Testing set (X): {}".format(X_test.shape))
                self.logger.info("Training set (y): {}".format(y_train.shape))
                self.logger.info("Testing set (y): {}".format(y_test.shape))
                # EDA
                train_eda = pd.concat([X_train, y_train], axis=1)
                train_eda.to_csv('eda.csv', index=False)

                #d = dtale.show(train_eda)
                #d.open_browser()
                #logging.info("EDA Completed, URL: {}".format(d._url))

            # Drop features with low variance
            X_train, X_test = self.low_variance_features(X_train, X_test)

            # Drop features with high correlation (Include categorical features also next)
            X_train, X_test = self.high_correlation_features(X_train, X_test, threshold=0.9)

            result = self.category_encode_dict(X_train)
            self.logger.info("Datatypes Result: {}".format(result))

            null_cols = X_train.columns[X_train.isnull().any()].tolist()  # this returns all columns having null columns
            impute_dict = self.column_datatypes(X_train, null_cols)
            self.logger.info("Features with Null Columns: {}".format(impute_dict))

            # Impute numerical null values
            if len(impute_dict['numerical']) > 0:
                only_numerical = impute_dict['numerical']
                X_train = self.numerical_missing_imputation(X_train, only_numerical, method="median")

            # Impute categorical null values
            if len(impute_dict['object']) > 0:
                categorical_dict = dict(X_train[impute_dict['object']].isnull().sum() * 100 / len(X_train))
                X_train = self.categorical_missing_imputation(X_train, categorical_dict)

            # Outlier Imputation
            if len(result['numerical']) > 0:
                outliers_list = result["numerical"]
                X_train = self.outlier_imputation(X_train, outliers_list)
                self.logger.info("Outlier Imputation done with Winsorizer")

            # Categorical Encoding
            if len(result['object']) > 0:
                encode_dict = {}
                for i in self.category_encode_dict(X_train)["object"]:
                    encode_dict[i] = len(list(pd.unique(X_train[i])))

                X_train, X_test = self.categorical_encoding(X_train, X_test, encode_dict)

            # CYCLICAL ENCODING
            if len(result['datetime']) > 0:
                date_to_use = ""
                if len(result['datetime']) > 1:
                    dict_count = dict(X_train[result['datetime']].isnull().sum())
                    min_key = min(dict_count, key=lambda k: dict_count[k])
                    date_to_use += min_key
                    date_to_use.strip()

                else:
                    date_to_use += result['datetime'][0]
                    date_to_use.strip()

                target_column = list(pd.DataFrame(y_train).columns)[0]

                Xy_train = pd.concat([X_train, y_train], axis=1)
                Xy_test = pd.concat([X_test, y_test], axis=1)
                X_train, X_test, y_train, y_test = self.cyclical_encoding(Xy_train, Xy_test, date_to_use, target_column)

                self.logger.info("Cyclical Encoding done using {} column".format(date_to_use))

            # Normalize data
            if len(result['numerical']) > 0:
                X_train, X_test = self.scale_features(X_train, X_test, result['numerical'])
                self.logger.info("Normalization done")

            # Label encoding Target variable
            if is_regression == False:
                y_train_encoded, y_test_encoded = self.encode_target_variable(y_train, y_test)
            else:
                 y_train_encoded, y_test_encoded = y_train, y_test

            self.logger.info("Training set (X): {}".format(X_train.shape))
            self.logger.info("Testing set (X): {}".format(X_test.shape))
            self.logger.info("Training set (y): {}".format(y_train_encoded.shape))
            self.logger.info("Testing set (y): {}".format(y_test_encoded.shape))


            # Save to CSV file
            self.df.to_csv(f'{self.name}', index=False)
            X_train.to_csv(f'X_train.csv', index=False)
            y_train_encoded.to_csv(f'y_train.csv', index=False)
            X_test.to_csv(f'X_test.csv', index=False)
            y_test_encoded.to_csv(f'y_test.csv', index=False)


            

            current_file_path = os.path.abspath(__file__)
            # Navigate to the parent directory (one level up)
            project_directory = os.path.dirname(current_file_path)

            # Now you're in the project directory
            print("Project directory:", project_directory)

         
            logging.info("Preprocessing Done!!!")
            print("Preprocessing Done!!!")


