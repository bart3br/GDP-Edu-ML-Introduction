from scrape_data import scrape_wikitable, pandas_scrape
import preprocessing as prep
import eda
import solution

def run_scraping() -> None:
    url_list = ['https://en.wikipedia.org/wiki/List_of_countries_by_tertiary_education_attainment',
                'https://en.wikipedia.org/wiki/List_of_countries_by_spending_on_education_(%25_of_GDP)',
                'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)_per_capita',
                'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population']
    
    #csv files downloaded manually:
    #   tertiary_edu_perent_over_the_years.csv ->
    #      from https://data.worldbank.org/indicator/SE.TER.CUAT.ST.MA.ZS?name_desc=true
    #   top1000_uni.csv ->
    #      from https://www.kaggle.com/datasets/mylesoneill/world-university-rankings
    
    scrape_wikitable(url_list[0], 'tertiary_edu_percent.csv', [(2,4)])
    pandas_scrape(url_list[1], 'percent_GDP_spent_on_edu.csv', 0)
    pandas_scrape(url_list[2], 'GDP_per_capita.csv', 1)
    pandas_scrape(url_list[3], 'population.csv', 1)
    
def run_preprocessing() -> None:
    #preprocessing
    population_df = prep.clean_population_data("population.csv")
    gdp_df = prep.clean_GDP_per_capita_data("GDP_per_capita.csv")
    uni_count_df = prep.count_universities("top1000_uni.csv", "country", 2015, "top1000_uni_count.csv")
    tertiary_df = prep.clean_tertiary_edu_percent_data("tertiary_edu_percent.csv")
    gdp_spent_df = prep.clean_percent_GDP_spent_data("percent_GDP_spent_on_edu.csv")
    tertiary_years_df = prep.clean_tertiary_edu_over_years_data("tertiary_edu_percent_over_the_years.csv")
    
    merged_df = prep.merge_data([population_df, gdp_df, uni_count_df, tertiary_df, gdp_spent_df], "Country")
    merged_df = prep.fill_NaN_expenditure_on_education(merged_df)
    merged_df = prep.fill_NaN_top1000_uni_count(merged_df)
    merged_df = prep.fill_NaN_tertiary_edu_percent(merged_df, tertiary_years_df)
    merged_df = prep.final_preprocessing(merged_df)
    merged_df.to_csv("dataset.csv", index=False)



def main():
    run_scraping()
    run_preprocessing()
    eda.run_EDA()
    solution.run_solution()    
    
if __name__ == "__main__":
    main()