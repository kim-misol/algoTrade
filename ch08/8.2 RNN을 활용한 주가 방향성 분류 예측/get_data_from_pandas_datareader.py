import pandas_datareader as pdr

# pandas_datareader 라이브러리를 통해 야후 파이낸스에서 데이터 가져오기
# BTC-KRW 데이터도 존재
intc_df = pdr.get_data_yahoo('INTC', start='2000-01-01')
sox_df = pdr.get_data_yahoo('^SOX', start='2000-01-01')
vix_df = pdr.get_data_yahoo('^VIX', start='2000-01-01')
snp500_df = pdr.get_data_yahoo('^GSPC', start='2000-01-01')
intc_df.to_csv('intc.csv')
sox_df.to_csv('sox_df.csv')
vix_df.to_csv('vix_df.csv')
snp500_df.to_csv('s&p500.csv')

