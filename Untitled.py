#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import xgboost as xgb


# In[2]:


# 데이터 불러오기
bike_data = pd.read_csv("_따릉이 10분 단위 잔여대수.csv", encoding = 'cp949')
location_data = pd.read_csv("_대여소 반경 400m 데이터.csv", encoding = 'cp949')
temp_data = pd.read_csv("_행정구역 및 시간별 기온.csv", encoding = 'cp949')
wind_data = pd.read_csv("_행정구역 및 시간별 풍속.csv", encoding = 'cp949')                         
air_data = pd.read_csv("_행정구역 및 일별 대기통합지수.csv", encoding = 'cp949')
rain_data = pd.read_csv("_행정구역 및 시간별 강수량.csv", encoding = 'cp949')
                        
master_data = pd.read_csv("_따릉이 대여소 마스터 정보.csv", encoding = 'cp949')
info = pd.read_csv("공공자전거 대여소 정보(23.06월 기준) (1).csv", encoding = 'cp949')


# In[3]:


# location_data : 열 이름 알파벳으로 바꾸기
location_data.rename(columns = {'# 지하철역' : 'subway_station', '# 공원' : 'park', '# 학교' : 'school', '# 문화시설' : 'cultural_facilities', '# 마트' : 'mart'}, inplace = True)
location_data


# In[4]:


#날짜 형식 맞추기

from datetime import datetime 

def format_column_name(column_name):
    try:
        parts = column_name.split()
        datetime_part = ' '.join(parts[:-1])
        parsed_datetime_part = datetime.strptime(datetime_part, '%Y.%m.%d %H:%M')
        return f'{parsed_datetime_part.strftime("%Y-%m-%d %H:%M")} {parts[-1]}'
    except:
        return column_name

bike_data.columns = [format_column_name(col) for col in bike_data.columns]


# In[13]:


## master_data와 info_data를 lat, long 기준으로 병함 => stationId와 guName을 merge
info_copy = info.copy()

info_copy = info_copy.rename(columns = {'Unnamed: 4' : 'stationLatitude', 'Unnamed: 5' : 'stationLongitude'})
info_copy.drop(columns = ['Unnamed: 3', '대여소\n번호', '설치\n시기', '설치형태', 'Unnamed: 8', '운영\n방식',
       'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',
       'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
       'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20'], inplace = True)

#결측치 제거
info_copy = info_copy.dropna()

#object 형태를 모두 float 형태로 변환
info_copy['stationLatitude'] = info_copy['stationLatitude'].astype(float)
info_copy['stationLongitude'] = info_copy['stationLongitude'].astype(float)

#info_data의 stationLatitude, stationLongitude를 모두 소수점 아래 6자리까지만 표현
info_copy = info_copy.round(6)

#info_data와 master_data 병합
merged_data_0 = pd.merge(master_data.round(6), info_copy, on=['stationLatitude', 'stationLongitude'])

# 소재지(위치) -> guName
merged_data_0.rename(columns = {'소재지(위치)' : 'guName'}, inplace = True)

# 보관소(대여소)명 열 제거
merged_data_0.drop(columns = ['보관소(대여소)명'], inplace = True)


# In[ ]:


import numpy as np
import random
from datetime import datetime

np.random.seed(42)
target_words = ['Temp', 'air', 'Rain', 'wind']

#새로운 데이터프레임 구성
new_df = pd.DataFrame(columns = ['t', 't-10', 't-20', 't-30', 't-40', 't-50', 't-60', 'Temp', 'air', 'Rain', 'wind', 'subway', 'cultural_facilities', 'school', 'park', 'mart'])

j = 0
for j in range(0,7) :
    #랜덤으로 시간대 선택

    random_date = pd.to_datetime(random.choice(pd.date_range(start="2023-12-04", end="2023-12-11")))

    random_time = pd.to_timedelta(str(random.choice(pd.date_range(start="2023-12-04", end="2023-12-11", freq='H')).time()))

    random_datetime = random_date + random_time
    random_datetime.strftime('%Y-%m-%d %H:%M')

    #랜덤으로 거치소 선택
    random_station = random.choice(bike_data.stationId)
    # stationId에 따라 guName 수집
    selected_guName = merged_data_0[merged_data_0['stationId'] == random_station]['guName']
    
    ##랜덤 시각, 랜덤 거치소에 대해 데이터프레임 구성하기
    # random_datetime 60분 전까지 있는지 확인

    time_columns = [random_datetime - pd.Timedelta(minutes=i*10) for i in range(0, 7)]

        
    ## 여기까지는 문제가 없어요!! 
    
    
    ##time_columns의 값이 모두 열 이름에 포함되어 있다면 각각을 t, ..., t-60에 저장 
    
    for bike_data_column in bike_data.columns :
        selected_row_1 = bike_data[bike_data['stationId'] == random_station]
        
        for i in range(7):
            matching_column = bike_data.columns[bike_data.columns.str.contains(time_column) & bike_data.columns.str.contains('parkingBike')]
            
            # 여기서 문제 발생 : 
            for time_column in time_columns :
                time_column = time_column.strftime('%Y-%m-%d %H:%M')
                if i == 0 :
                    new_df['t'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 1 :
                    new_df['t-10'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 2 :
                    new_df['t-20'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 3 :
                    new_df['t-30'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 4 :
                    new_df['t-40'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 5 :
                    new_df['t-50'] = selected_row_1[matching_column].stack().reset_index(drop = True)
                if i == 6 :
                    new_df['t-60'] = selected_row_1[matching_column].stack().reset_index(drop = True)          
                
                #air, Temp, Rain, wind _data에서 selected_guName에 따라 random_datetime 열 수집 
                selected_air = air_data[air_data['guName'].isin(selected_guName)]
                for air_data_column in selected_air.columns :
                    if str(random_datetime) in air_data_column :
                        new_df['air'].extend(selected_air[air_data_column].values)
                    else :
                        continue
                
                selected_rain = rain_data[rain_data['guName'].isin(selected_guName)]
                for rain_data_column in selected_rain.columns :
                    if str(random_datetime) in rain_data_column :
                        new_df['Rain'].extend(selected_rain[rain_data_column].values)
                    else :
                        continue
                
                selected_wind = wind_data[wind_data['guName'].isin(selected_guName)]
                for wind_data_column in selected_wind.columns :
                    if str(random_datetime) in wind_data_column :
                        new_df['wind'].extend(selected_wind[wind_data_column].values)      
                    else :
                        continue
                
                selected_temp = temp_data[temp_data['guName'].isin(selected_guName)]
                for temp_data_column in selected_temp.columns :
                    if str(random_datetime) in temp_data_column :
                        new_df['Temp'].extend(selected_temp[temp_data_column].values)  
                    else :
                        continue
                
                #시설 데이터 저장
                selected_row_2 = location_data[location_data['stationId'] == random_station]
                new_df['subway_station'] = selected_row_2['subway_station'].values
                new_df['mart'] = selected_row_2['mart'].values
                new_df['park'] = selected_row_2['park'].values
                new_df['cultural_facilities'] = selected_row_2['cultural_facilities'].values
                new_df['school'] = selected_row_2['school'].values

                print(new_df)
                j = j+1
        else : 
            j = j+1
            continue
            


# In[107]:


for time_column in time_columns :
    time_column= time_column.strftime('%Y-%m-%d %H:%M')
    #time_columns[1] = time_column[1].strftime('%Y-%m-%d %H:%M')
    
    for selected_col in selected_row_1.columns :
        if str(time_column) in selected_col_name :
    
    new_df['t'] = selected_row_1[time_column].values
    new_df


# In[ ]:





# In[ ]:





# In[ ]:




