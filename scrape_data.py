import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_wikitable(url: str, filename: str, subheader_tuple_list: list[tuple] = [], custom_data_start_row: int = 0) -> pd.DataFrame:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    #getting the table with beautifulsoup, here we are searching for 'wikitable' class, which represents the table on wikipedia
    #here with BS we are able to find "wikitable sortable jquery-tablesorter" class
    table = soup.find('table', {'class':'wikitable sortable'})
    
    headers = [] #list of table headers
    rows = table.findAll('tr') #getting all table rows
    
    #getting all table headers, if there are no subheaders, we get the headers from the first row 
    #otherwise we use the function fix_subheader_rows to merge the subheaders with the headers
    if subheader_tuple_list == []:
        for cell in rows[0].findAll(['th', 'td']):
            headers.append(cell.text.strip())
    else:
        headers = fix_subheader_rows(rows[0], rows[1], subheader_tuple_list)
    
    #data dictionary for collecting all data, here we prepare the dictionary with empty lists for each header
    data = {}
    for header in headers:
        data[header] = []
    
    #starting row is first row with data, if there are no subheaders, it is the 1st row, otherwise it's the 2nd row    
    starting_row = 1 if subheader_tuple_list == [] else 2
    
    for row in rows[starting_row:]:
        cells = row.findAll('td')
        for indx, cell in enumerate(cells):
            data[headers[indx]].append(cell.text.strip())
            
    df = pd.DataFrame(data)
    df.to_csv(filename, na_rep='-', index=False, encoding='utf-8')
    return df

def fix_subheader_rows(first_row, second_row, split_header_tuple_list: list[tuple]) -> list:
    """Merges subheader rows with the header row, and returns the new header row.
       split_header_tuple_list is a list of tuples, where tuple[0] is the index of the header containing subheaders,
       and tuple[1] is the number of subheaders in that header."""
    
    headers = []
    frst_header_cells = first_row.findAll(['th', 'td'])
    scnd_header_cells = second_row.findAll(['th', 'td'])
    flag = False
    sub_header_count = 0
    
    for indx in range(len(frst_header_cells)):
        flag = False
        for tup in split_header_tuple_list:
            if indx == tup[0]:
                flag = True
                for i in range(sub_header_count, sub_header_count + tup[1]):
                    headers.append(frst_header_cells[indx].text.strip() + ' ' + scnd_header_cells[i].text.strip())
                    sub_header_count += 1
        if not flag:
            headers.append(frst_header_cells[indx].text.strip())
    
    return headers

    
def pandas_scrape(url:str, filename: str, table_index:int):
    df = pd.read_html(url)
    df[table_index].to_csv(filename, na_rep='-', index=False, encoding='utf-8')
    return df[table_index]

if __name__ == "__main__":
    pass