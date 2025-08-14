OPTIONS (SKIP=1)
LOAD DATA
INFILE 'C:\\Users\\ASUS\\Downloads\\penta\\MOCK_DATA.csv'
INTO TABLE survey_data
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
TRAILING NULLCOLS
(
  id,
  occupation,
  state,
  district,
  gender,
  income,
  year
)
