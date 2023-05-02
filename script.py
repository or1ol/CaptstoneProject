import sys
import os

month_names = ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']
months = range(1,13)
i2m = list(zip(months, month_names))

def download_data(year, dataset, link):
    
    print("Current working directory before")
    cwd = os.getcwd()
    print(cwd)
    print()
    os.system(f"mkdir -p dades/{year}/{dataset}")
    # trying to insert to false directory
    try:
        print(f"Working at-{cwd}/dades/{year}/{dataset}")
        os.chdir(f"{cwd}/dades/{year}/{dataset}")
        
        for month, month_name in i2m:
            os.system(f"wget '{link}/{year}_{month:02d}_{month_name}_{dataset}.7z'")
            os.system(f"7z x '{year}_{month:02d}_{month_name}_{dataset}.7z'")
            os.system(f"rm '{year}_{month:02d}_{month_name}_{dataset}.7z'")
            if month_name == 'Marc':
                os.system(f"mv '{cwd}/dades/{year}/{dataset}/{year}_{month:02d}_Març_{dataset}.csv' '{cwd}/dades/{year}/{dataset}/{year}_{month:02d}_{month_name}_{dataset}.csv'")
        
        if year == '2019':
            print('correcting file names')
            for month, month_name in i2m[2:5]:
                if month_name == 'Marc':
                    month_name = 'Març'
                
                if dataset=='BicingNou_ESTACIONS':
                    tochange='STAT'
                elif dataset=='BicingNou_INFORMACIO':
                    tochange='INFO'
                else:
                    break
                
                os.system(f"mv '{cwd}/dades/{year}/{dataset}/{year}_{month:02d}_{month_name}_BICING2_{tochange}.csv' '{cwd}/dades/{year}/{dataset}/{year}_{month:02d}_{month_name}_{dataset}.csv'")
        
        print(f"Done at-{cwd}/dades/{year}/{dataset}")
    # Caching the exception    
    except:
        print("Something wrong with specified directory. Exception- ")
        print(sys.exc_info())

    # handling with finally          
    finally:
        print()
        print("Restoring the path")
        os.chdir(cwd)
        print("Current directory is-", cwd)

if __name__ == "__main__":
    year = sys.argv[1]
    dataset = sys.argv[2]
    link = sys.argv[3]
    
    download_data(year,dataset,link)