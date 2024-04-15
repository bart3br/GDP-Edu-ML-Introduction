import pandas as pd
import numpy as np
import random


def count_universities(input_file: str, key_column: str, year: int, output_file: str) -> pd.DataFrame:
    """This method counts the number of universities of each country in the given csv file.
       The result is returned in pandas dataframe and saved to csv file."""
    try:
        input_df = pd.read_csv(input_file)
        filtered_df = input_df[input_df["year"] == year]
        grouped_df = filtered_df.groupby(key_column).size().reset_index(name='Top_1000_Uni_Count')
        grouped_df = grouped_df.rename({'country': 'Country'}, axis=1)
        grouped_df.loc[grouped_df['Country'] == 'USA', 'Country'] = 'United States'
        grouped_df.to_csv(output_file, index=False)
        return grouped_df
        
    except FileNotFoundError:
        print(f"Couldn't open {input_file}. File not found.")
        return None

def count_avg(nums: list) -> float:
    """This method counts the average of given list of numbers (checks for NaN values)"""
    avg = 0
    div_count = 0
    null_vals = ["NaN", "-", " ", "—", "NULL", "nan", "n.a.", np.nan]
    
    for i in range(len(nums)):
        if nums[i] not in null_vals:
            avg += float(nums[i])
            div_count += 1
    
    return float(avg/div_count) if div_count != 0.0 else 0.0
            
            
        
def clean_GDP_per_capita_data(filename:str) -> pd.DataFrame:
    """This method is used to clean the GDP_per_capita csv file
       Making one column for the country name, one column for the GDP per capita
       And one column for the UN region
       GDP per capita is from different sources and years, so the average is counted
       Deleting unnecessary columns"""
    try:
        df = pd.read_csv(filename)
        df = df.drop(0) #drop first of two rows with headers
        df = df.drop(1) #drop first row of data, which is empty
        df = df.rename({'Country/Territory': 'Country', 'UN Region': 'UN_Region'}, axis=1)
        #now we need to add the 'average GDP per capita' column
        num_of_rows = len(df.index)
        avgGDPPC = []
        for i in range(num_of_rows):
            avgGDPPC.append(int(count_avg([df.iloc[i, 2], df.iloc[i, 4], df.iloc[i, 6]])))
        df['GDP_per_capita'] = avgGDPPC
        
        #deleting the columns with the GDP per capita from different sources and years
        df = df.drop(df.columns[[2,3,4,5,6,7]], axis=1)
        
        df.reset_index(drop=True, inplace=True)
        return df
        
    except FileNotFoundError:
        print(f"Couldn't open {filename}. File not found.")
        return None

def clean_percent_GDP_spent_data(filename: str) -> pd.DataFrame:
    """This method is used to clean the percent_GDP_spent_on_edu csv file
       Making one column for the country name and one column for the expenditure on education
       Deleting unnecessary columns"""
    try:
        df = pd.read_csv(filename)
        
        #renaming country column so it can be merged with other dataframes
        df = df.rename({'Country or subnational area': 'Country',
                        'Expenditure on education (% of GDP)': 'Expenditure_on_education_(%_of_GDP)'}, axis=1)
        #deleting the columns with year and source
        df = df.drop(df.columns[[2,3]], axis=1)
        
        df.reset_index(drop=True, inplace=True)        
        return df
        
    except FileNotFoundError:
        print(f"Couldn't open {filename}. File not found.")
        return None
    
def clean_population_data(filename: str) -> pd.DataFrame:
    """This method is used to clean the population csv file
       Making one column for the country name and one column for population
       Deleting unnecessary columns"""
    try:
        df = pd.read_csv(filename)
        df = df.drop(0)
        
        df = df[df['Rank'] != '–']
        
        #renaming country column so it can be merged with other dataframes
        df = df.rename({'Country / Dependency': 'Country', 'Numbers': 'Population'}, axis=1)
        #deleting the columns with year and source
        df = df.drop(df.columns[[0,3,4,5,6]], axis=1)
        
        df.reset_index(drop=True, inplace=True)
        return df
        
    except FileNotFoundError:
        print(f"Couldn't open {filename}. File not found.")
        return None

def clean_tertiary_edu_percent_data(filename: str) -> pd.DataFrame:
    """This method is used to clean the tertiary_edu_percent csv file
       Making one column for the country name and one column for the tertiary education percentage
       Deleting unnecessary columns"""
    try:
        df = pd.read_csv(filename)
        
        #renaming percent column so the name refers to the data it contains
        df = df.rename({'Age25–64(%)': 'Tertiary_edu_%'}, axis=1)
        #deleting the columns with year and source
        df = df.drop(df.columns[[2,3,4,5,6,7]], axis=1)
        
        df.reset_index(drop=True, inplace=True)
        return df
        
    except FileNotFoundError:
        print(f"Couldn't open {filename}. File not found.")
        return None
    
def clean_tertiary_edu_over_years_data(filename:str) -> pd.DataFrame:
    """This method is used to clean the tertiary_edu_percent_over_the_years csv file
       Making one column for the country name and one column for the tertiary education percentage
       There is historical data from many years for each country, so the newest data is chosen
       Deleting unnecessary columns"""
    try:
        df = pd.read_csv(filename)
        #for i in range(4):
         #   df = df.drop(0) #drop first 4 rows (5th row is the header row)
            
        df = df.rename({'Country Name': 'Country'}, axis=1)
        
        tertiary_columns = df.columns[4:]
        null_vals = ["NaN", "-", " ", "—", "NULL", "nan", "n.a.", np.nan]
        newestData = []
        #for every country finding the newest value of the tertiary education percentage
        for index, row in df.iterrows():
            newest_val = None
            for col in tertiary_columns:
                if str(row[col]) not in null_vals:
                    newest_val = row[col]
            newestData.append(round(newest_val,1)) if newest_val is not None else newestData.append(np.nan)
       
        
        #deleting the columns with the GDP per capita from different sources and years
        df = df.drop(df.columns[1:], axis=1)
        
        df['Tertiary_edu_%'] = newestData
        df.reset_index(drop=True, inplace=True)
        return df
        
    except FileNotFoundError:
        print(f"Couldn't open {filename}. File not found.")
        return None


            
def merge_data(input_df_list: list[pd.DataFrame], merge_col: str) -> pd.DataFrame:
    """
    This method is used to merge csv files into one csv file
    input_file_list: list of csv files to be merged
    merge_col: name of the column that will be used to merge the files
    
    first element of the input_file_list is the file that will be used as the base for the merge
    left merge is used here, so the data from the first file will be preserved"""
    if len(input_df_list) < 2:
        return None
    
    merged_df = None
    
    for indx in range(len(input_df_list)):
        df = input_df_list[indx]
        if indx == 0 or merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=merge_col, how='left')
            
    return merged_df


#filling NaN values
def fill_NaN_top1000_uni_count(df: pd.DataFrame) -> pd.DataFrame:
    """This method is used to fill the NaN values in the Top_1000_Uni_Count column with 0
       It's filled with 0 because the NaN values mean,
       that the country doesn't have any universities in the top 1000"""
       
    df['Top_1000_Uni_Count'].fillna(0, inplace=True)
    return df

def fill_NaN_expenditure_on_education(df: pd.DataFrame) -> pd.DataFrame:
    """This method is used to fill the NaN values in the Expenditure_on_education_(%_of_GDP) column
       with the average expenditure on education for the country's region
       The average expenditure is calculated from the data that is available
       And is then multiplied by a random number between 0.8 and 1.2"""
       
    df['Expenditure_on_education_(%_of_GDP)'].replace('n.a.', np.nan, inplace=True)
    df['Expenditure_on_education_(%_of_GDP)'] = pd.to_numeric(df['Expenditure_on_education_(%_of_GDP)'], errors='coerce')
    
    avg_expenditure = df.groupby('UN_Region')['Expenditure_on_education_(%_of_GDP)'].mean()
    
    #setting the seed for reproducibility
    random.seed(33)
    #adding random noise to the average expenditure
    new_avg_expenditure = avg_expenditure.apply(lambda x: round(x*(1+random.uniform(-0.2, 0.2)), 1))
    
    df['Expenditure_on_education_(%_of_GDP)'].fillna(df['UN_Region'].map(new_avg_expenditure), inplace=True)
    return df
    
    
def fill_NaN_tertiary_edu_percent(df: pd.DataFrame, wb_df: pd.DataFrame) -> pd.DataFrame:
    """This method is used to fill NaN values in the Tertiary_edu_% column
       It takes existing merged dataframe as an input (it already contains the Tertiary_edu_% column)
       Then fills missing values with data from wb_df dataframe (source: World Bank)
       If there are still any reamining NaN values, it fills them with the avg value for the UN region +/- random noise"""
    
    df.reset_index(drop=True, inplace=True)
    wb_df.reset_index(drop=True, inplace=True)
    
    #merging the dataframes by adding new temporary column
    wb_df = wb_df.rename({'Tertiary_edu_%': 'temp_tertiary'}, axis=1)
    merged_df = pd.merge(df, wb_df, on='Country', how='left')
    
    #filling NaN values in tertiary education column with data from World Bank, then dropping the temporary column
    merged_df['Tertiary_edu_%'].fillna(merged_df['temp_tertiary'], inplace=True)
    merged_df.drop('temp_tertiary', axis=1, inplace=True)
    
    avg_tertiary = merged_df.groupby('UN_Region')['Tertiary_edu_%'].mean()
    
    #setting the seed for reproducibility
    random.seed(33)
    #adding random noise to the average tertiary education percentage
    new_avg_tertiary = avg_tertiary.apply(lambda x: round(x*(1+random.uniform(-0.2, 0.2)), 1))
    
    #filling remaining NaN values with the avg tertiary education percentage for the UN region
    merged_df['Tertiary_edu_%'].fillna(merged_df['UN_Region'].map(new_avg_tertiary), inplace=True)
    return merged_df

def final_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """This method is used to drop rows with NaN values in the UN_Region column
       It also converts the Top_1000_Uni_Count column to int type
       And sorts the rows by country name, so it's easier to navigate through the dataframe"""
       
    df.dropna(subset=['UN_Region'], inplace=True)
    df['Top_1000_Uni_Count'] = df['Top_1000_Uni_Count'].astype(int)
    df.sort_values(by=['Country'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    pass