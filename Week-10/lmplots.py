import pandas as pd
import seaborn as sns
employee = pd.read_csv("employee.csv", parse_dates=['HIRE_DATE',
                                                    'JOB_DATE'])
days_hired = pd.to_datetime('12-1-2016') - employee['HIRE_DATE']
one_year = pd.Timedelta(1, unit='Y')
employee['YEARS_EXPERIENCE'] = days_hired / one_year

print(employee[['HIRE_DATE', 'YEARS_EXPERIENCE']].head())
