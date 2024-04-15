import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import ydata_profiling as pdp

def distribution(df, x, title, xlabel, figsize=(10, 5)):
    """distribution plot of a column in a dataframe
       in a form of histogram with density plot
       plots are saved in plots folder"""
       
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90)
    
    #filtering data to remove ouliers for better visualization
    filtered_data = df[(df[x] > df[x].quantile(0.05)) & (df[x] < df[x].quantile(0.95))]
    
    sns.histplot(filtered_data[x], stat='density', kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    
    plt.savefig(f"plots/histplot_{xlabel}.png")
    plt.close()
    
def run_EDA():
    """exploratory data analysis for the dataset
       creating a pandas profiling report using ydata_profiling (substitute for pandas_profiling in python 11)
       the report shows information about the dataset (each column), correlations between columns, and more
       then creating distribution plots for each column in the dataset (distribution method), 
       excluding 'Country' and 'UN_Region'"""
    
    df = pd.read_csv("dataset.csv")
    print(df.sample(10))
    print(df.info())
    report = pdp.ProfileReport(df)
    report.to_file("ydata_profiling/prof_report.html")
    
    column_names = ['Population', 'GDP_per_capita', 'Top_1000_Uni_Count', 'Tertiary_edu_%', 
                    'Expenditure_on_education_(%_of_GDP)']
    distribution(df, column_names[0], f"Distribution of {column_names[0]}", column_names[0])
    distribution(df, column_names[1], f"Distribution of {column_names[1]}", column_names[1])
    distribution(df, column_names[2], f"Distribution of {column_names[2]}", column_names[2])
    distribution(df, column_names[3], f"Distribution of {column_names[3]}", column_names[3])
    distribution(df, column_names[4], f"Distribution of {column_names[4]}", column_names[4])
    
    
if __name__ == "__main__":
    pass