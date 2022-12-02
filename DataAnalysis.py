import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
import scipy        
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm    
import seaborn as sns
import itertools as it
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
#from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller


## Sales 2018-2021/Jul ##
sales_2018 = pd.read_csv("\\sales_2018-01-01_2018-12-31.csv")
sales_2019 = pd.read_csv("\\sales_2019-01-01_2019-12-31.csv")
sales_2020 = pd.read_csv(\\sales_2020-01-01.csv")
sales_2021 = pd.read_csv("\\sales_2021-01-01_2021-08-04.csv")
    ## remove August
    sales_2021['month'] = pd.to_datetime(sales_2021['month'])
    sales_2021 = sales_2021[sales_2021['month'].dt.month != 8]

sales_object = [sales_2018,sales_2019,sales_2020,sales_2021]
sales = pd.concat(sales_object)

sales.columns
sales.head(3)
sales.info()
sales.describe()
for i in range(sales.shape[1]):
    print(sales.iloc[:,i].describe() )
    print("")

for i in range(sales.shape[1]):
    print(sales.columns[i])
    print(sales.iloc[:,i].dtype )
    print("")

## Transforming ['month'] to datetime type
sales['month'] = pd.to_datetime(sales['month'])
print(sales['month'].dtype)
sales['month'].unique()

## How many different products have been sold
sales.groupby('month')['product_type'].nunique() 

## HeatMap
grouped_total_net_sales = pd.DataFrame(sales.groupby(['month','product_type'],sort=True)['net_sales'].sum()).reset_index()
grouped_total_net_sales_pivot = grouped_total_net_sales.pivot(values='net_sales',index='month', columns='product_type'); print(grouped_total_net_sales_pivot)
grouped_total_net_sales.groupby('month')['net_sales'].sum().plot(title='Total Sales')


## slicing key metrics
sale_type = sales[['month','product_type','total_sales','net_sales','gross_profit','gross_margin','billing_region','billing_country']]
sale_type.set_index('month')


## TOTAL SALES : How much of each product has been sold
grouped_total_sales = pd.DataFrame(sale_type.groupby(['month','product_type'],sort=True)['total_sales'].sum()).reset_index()
grouped_total_sales = grouped_total_sales.set_index('product_type')

grouped_total_sales.groupby(['month'])['total_sales'].nlargest(1) # Top Product by Month

top_total_sales = pd.DataFrame(grouped_total_sales.groupby(['month'])['total_sales'].nlargest(3)).reset_index()
top_total_sales.set_index('month') # Top 3 products by month
top_total_sales['product_type'].value_counts(sort=True) # summary of top products sales
top_total_sales.groupby('product_type').sum().sort_values('total_sales',ascending=False)


### NET SALES : How much of each product has been sold
grouped_net_sales = pd.DataFrame(sale_type.groupby(['month','product_type'],sort=True)['net_sales'].sum()).reset_index()
grouped_net_sales = grouped_net_sales.set_index('product_type')

grouped_net_sales.groupby(['month'])['net_sales'].nlargest(1) # Top Product by Month

top_net_sales = pd.DataFrame(grouped_net_sales.groupby(['month'])['net_sales'].nlargest(3)).reset_index()
top_net_sales.set_index('month') # Top 3 products by month
top_net_sales['product_type'].value_counts(sort=True) # summary of top products net sales
top_net_sales.groupby('product_type').sum().sort_values('net_sales',ascending=False)
    top_products_sale = top_net_sales.groupby('product_type').sum().sort_values('net_sales',ascending=False).index.values[:3]

products_T1 = top_net_sales.groupby('product_type').sum().sort_values('net_sales',ascending=False).index.values[:5]
products_T2 = top_net_sales.groupby('product_type').sum().sort_values('net_sales',ascending=False).index.values[5:10]


 # PLOT monthly sales
grouped_net_sales = pd.DataFrame(sale_type.groupby(['month','product_type'],sort=True)['net_sales'].sum()).reset_index()
df1 = grouped_net_sales[grouped_net_sales['product_type'].isin(products_T1)].reset_index(drop=True)
df2 = grouped_net_sales[grouped_net_sales['product_type'].isin(products_T2)].reset_index(drop=True)
sales_plot_T1 = df1.pivot(values='net_sales',index='month', columns='product_type'); print(sales_plot_T1)
sales_plot_T2 = df2.pivot(values='net_sales',index='month', columns='product_type'); print(sales_plot_T2)
sales_plot_T1.plot() #colormap='Set3'  #color=['red','orange','green','blue']
sales_plot_T2.plot() # colormap='Set2' #color=['red','orange','green','blue']
    
    # Plot cumulative sales
cum_sales_plot_T1 = sales_plot_T1.cumsum(); print(cum_sales_plot_T1)
cum_sales_plot_T2 = sales_plot_T2.cumsum(); print(cum_sales_plot_T2)
cum_sales_plot_T1.plot() 
cum_sales_plot_T2.plot() 

    # Plot shared y-axis Product Cumulative Sales
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
cum_sales_plot_T1.plot(ax=ax1)
cum_sales_plot_T2.plot(ax=ax2)

    # Plot Correlation Heatmap
cor_products_T1 = sales_plot_T1.corr(method='pearson'); print(cor_products_T1)
sns.heatmap(cor_products_T1, cmap='PiYG',annot=True,vmin=-1,vmax=1)    

cor_products_T2 = sales_plot_T2.corr(method='pearson'); print(cor_products_T2)
sns.heatmap(cor_products_T2, cmap='PiYG',annot=True,vmin=-1,vmax=1)    

## Pct Return Analysis (vs Jan/2021)
total_cum_sales = grouped_net_sales.pivot(values='net_sales',index='month', columns='product_type').cumsum(); print(total_cum_sales)

    # Dollar-wise
growth = total_cum_sales.iloc[-1,:] - total_cum_sales.iloc[-7,:]
top_growth = growth.sort_values(ascending=False)
print(top_growth.round(1)[:15])

    top_growth.round(1)[:15].plot(kind='bar',title='Accum Sales Growth YTD: Top Products',color='blue')
    growth_products = top_growth.index.to_list()[:15]
    total_cum_sales[growth_products[:5]][-7:].plot()
    total_cum_sales[growth_products[5:10]][-7:].plot()
    total_cum_sales[growth_products[10:15]][-7:].plot()
    
    # percentage-wisw
returns = total_cum_sales.pct_change(periods=6).tail(1); print(returns)
top_returns = returns.T.sort_values('2021-07-01',ascending=False)*100
top_returns = top_returns.rename({top_returns.columns[0]:'Jul vs Jan (accum)'},axis='columns',errors='raise'); print(top_returns)
top_returns[top_returns.iloc[:,0]>0].round(1)[:20] # Top 20

    return_products = top_returns.index.to_list()[:10]
    total_cum_sales[return_products[:5]][-7:].plot()
    total_cum_sales[return_products[5:]][-7:].plot()

grouped_net_sales_pivoted = grouped_net_sales.pivot(values='net_sales',index='month', columns='product_type')
growth_sales = grouped_net_sales_pivoted.pct_change(periods=6).tail(1).T.sort_values(by=grouped_net_sales['month'][-1:].values[0],ascending=False)*100
growth_sales = growth_sales.rename({growth_sales.columns[0]:'Jul vs Jan'},axis='columns',errors='raise')
print(growth_sales.round(1)[:20])


#### TOTAL SALES
total_sales = pd.DataFrame(sale_type.groupby(['month'],sort=True)['net_sales'].sum()).reset_index()
total_sales = total_sales.set_index('month')
total_sales.plot(color='blue', title='Total Net Sales')

total_cum_sales = total_sales.cumsum()
total_cum_sales.plot(color='red', title='Total Accumulated Net Sales')

total_cum_sales.pct_change(periods=1).rolling(window=6).mean().plot(color='orange',title = 'Monthly %-Change in Sales (6-month avg)')
total_cum_sales.diff(1).rolling(window=6).mean().plot(color='green',title = 'Monthly Dollar-Change in Sales (6-month avg)')


    ### SALES NOT INCLUDING PANDEMIC PRODUCTSL Neck Gaiter and Mask With Nose Shape
    pandemic_product1 = pd.DataFrame(grouped_net_sales_pivoted['Neck Gaiter'] ).fillna(0)
    pandemic_product2 = pd.DataFrame(grouped_net_sales_pivoted['Mask With Nose Shape']).fillna(0)
    
    df_ = (total_sales.join(pandemic_product1)).join(pandemic_product2)
    total_sales_adj = pd.DataFrame(df_.loc[:,'net_sales'] - df_.loc[:,'Neck Gaiter'] - df_.loc[:,'Mask With Nose Shape'],columns=['net_sales_adj'])
    
    total_sales_adj.join(total_sales).plot(color=['navy', 'blue'])



### Time Series Analysis
np.log(total_sales_adj).plot(title='log') # Log
total_sales_adj.diff().plot(title='diff'); # Diff
np.log(total_sales_adj).diff().plot(title='log-diff'); # Diff of Log
    
tsaplots.plot_pacf(np.log(total_sales_adj), lags=5); 
tsaplots.plot_pacf(total_sales_adj.diff().fillna(0), lags=5); 


tsm = ARMA(np.log(total_sales_adj), order=(1,0))
tsm.fit(trend='c').summary()


        ## Detrending
        plt.plot(np.linspace(1,13).reshape(-1,1),reg.predict(np.linspace(1,13).reshape(-1,1)))

        from scipy import signal
        
        detrended = signal.detrend(np.log(total_sales_adj.values),type='linear')
        plt.plot(detrended )
    
        ## Detrending
        detrended = np.log(total_sales_adj) - decomp_trend
        np.log(total_sales_adj).plot(); detrended.plot()




## Decomposition
rcParams['figure.figsize'] = 11, 9
   
   ### Net Sales Adjusted
   decomp=sm.tsa.seasonal_decompose(np.log(total_sales_adj),model='additive',extrapolate_trend='freq'); decomp.plot()    
   ## Trend 
   decomp_trend = decomp.trend.values.reshape(-1,1)
        ax = (pd.DataFrame(decomp_trend).diff(1)).plot(color="blue",title='chg in log-trend')
        ax.set_ylabel('chg in log-trend')
        trend_chg_mean = (pd.DataFrame(decomp_trend).diff(1)).mean()
        ax.axhline(trend_chg_mean.values,color='green', linestyle='--')
        plt.legend(['chg in log-trend','mean'])
        plt.show()
        
        ax = (pd.DataFrame(np.exp(decomp_trend)).pct_change(1)*100).plot(color="red",title='% chg in organic growth')
        ax.set_ylabel('% chg')
        trend_chg_mean = (pd.DataFrame(np.exp(decomp_trend)).pct_change(1)*100).mean()
        ax.axhline(trend_chg_mean.values,color='orange', linestyle='--')
        plt.legend(['% chg in trend','mean'])
        plt.show()

    t = np.arange(1,len(total_sales_adj)+1)
    trend_model = OLS(decomp_trend,t.reshape(-1,1)).fit()
    trend_model.summary()
        trend_df = pd.DataFrame(decomp_trend,index=total_sales_adj.index,columns=['decomp_trend'])
        trend_df.join(pd.Series(trend_model.predict(t),name='OLS_pred',index=total_sales_adj.index) )
        trend_model.predict(t).plot()
    
    c=(decomp_trend[-1:]-decomp_trend[0])/len(t)
    fv=c*t+12
    plt.plot(t,decomp_trend); plt.plot(t,fv.reshape(-1,1))

    ## Seasonal
    decomp.seasonal
    tsaplots.plot_pacf(decomp.seasonal, lags=13); 
    tsaplots.plot_acf(decomp.seasonal, lags=13);

        (np.exp(decomp.seasonal).pct_change(1)*100)[:13]
        decomp_seasonal = np.exp(decomp.seasonal).pct_change(1)*100
        ax = decomp_seasonal[:13].plot(color="red",title='% chg in seasonal component')
        ax.set_ylabel('% chg')
        plt.show()
    
    seasonal_fcst = decomp.seasonal[-12:]
    
    
    ## Residual
    decomp.resid
    adf =  pd.DataFrame(adfuller(decomp.resid,regression='c'),index=['adf', 'pvalue', 'usedlag', 'nobs', 'critical_values', 'icbest'])
         print(adf)
         np.mean(decomp.resid)
         np.std(decomp.resid)
    plt.hist(decomp.resid,alpha=.9,normed=True)
    ax = sns.distplot(decomp.resid); ax.axvline(x=0,linestyle='--')
    jb = stats.jarque_bera(decomp.resid); print(jb.pvalue)
    
    
        ## forecasting Noise via Monte Carlo simulation
        from scipy.stats import norm    
        #z = stats.norm.rvs(loc=0,scale=np.std(decomp.resid),size=1000) # 
        resid_fcst = norm.ppf(np.random.rand(12, 1000))


   ### Net Sales (no adjustment)
   decomp_=sm.tsa.seasonal_decompose(np.log(total_sales),model='additive',extrapolate_trend='freq'); decomp_.plot() 

   ## Trend 
   decomp_trend_ = decomp_.trend.values.reshape(-1,1)
       
        ax = (pd.DataFrame(np.exp(decomp_trend_)).pct_change(1)*100).plot(color="red",title='% chg in organic growth')
        ax.set_ylabel('% chg')
        trend_chg_mean_ = (pd.DataFrame(np.exp(decomp_trend_)).pct_change(1)*100).mean()
        ax.axhline(trend_chg_mean_.values,color='orange', linestyle='--')
        plt.legend(['% chg in trend','mean'])
        plt.show()
  

    ## Seasonal
    decomp_.seasonal
    tsaplots.plot_pacf(decomp_.seasonal, lags=13); 
    tsaplots.plot_acf(decomp_.seasonal, lags=13);

        (np.exp(decomp_.seasonal).pct_change(1)*100)[:13]
        decomp_seasonal_ = np.exp(decomp_.seasonal).pct_change(1)*100
        ax = decomp_seasonal_[:13].plot(color="red",title='% chg in seasonal component')
        ax.set_ylabel('% chg')
        plt.show()
        
        decomp_notLog = sm.tsa.seasonal_decompose(total_sales,model='additive',extrapolate_trend='freq');
            decomp_notLog.plot() 
        (decomp_notLog.seasonal[:13]).plot(title = 'Seasonal Component'); plt.ylabel( '$ Sales')
        
        decomp_seasonal_notLog = (decomp_notLog.seasonal-decomp_notLog.seasonal[0])/decomp_notLog.seasonal[0]*-100
        ax = decomp_seasonal_notLog[:13].plot(color="red",title='% chg in seasonal component')
        ax.set_ylabel('% chg')
        plt.show()
    
    
    seasonal_fcst_ = decomp_.seasonal[-12:]
    
    ## Residual
    decomp_.resid
    adf_ =  pd.DataFrame(adfuller(decomp_.resid,regression='c'),index=['adf', 'pvalue', 'usedlag', 'nobs', 'critical_values', 'icbest'])
         print(adf_)
         np.mean(decomp_.resid)
         np.std(decomp_.resid)
    plt.hist(decomp_.resid,alpha=.9,normed=True)
    ax = sns.distplot(decomp_.resid); ax.axvline(x=0,linestyle='--')
    
    jb_ = stats.jarque_bera(decomp_.resid); print(jb.pvalue)
    
    
        ## forecasting Noise via Monte Carlo simulation
        #z = stats.norm.rvs(loc=0,scale=np.std(decomp.resid),size=1000) # 
        resid_fcst_ = norm.ppf(np.random.rand(12, 1000))    




#### Forecasting
pip install pystan==2.19.1.1
pip install prophet
conda install -c conda-forge prophet
py -m pip install ./downloads/prophet-1.0.1.tar.gz
from prophet import Prophet

total_sales_adj_ = total_sales_adj.reset_index()
total_sales_adj_ = total_sales_adj_.rename(columns={'month':'ds', 'net_sales_adj':'y'})
prophet_model_ = Prophet(changepoint_prior_scale=0.15,growth='linear',yearly_seasonality=True,seasonality_mode='multiplicative')
prophet_model_.fit(total_sales_adj_)

prophet_forecast_ = prophet_model_.make_future_dataframe(periods=12, freq='M')
prophet_forecast_ = prophet_model_.predict(prophet_forecast_)

prophet_model_.plot(prophet_forecast_,xlabel='',ylabel='sales'); plt.show()

prophet_model_.changepoints

print(prophet_model_)

    ## using actual Sales time series
    total_sales_ = total_sales.reset_index()
    total_sales_= total_sales_.rename(columns={'month':'ds', 'net_sales':'y'})
    prophet_model = Prophet(changepoint_prior_scale=0.5)
    prophet_model.fit(total_sales_)
    
    prophet_forecast = prophet_model.make_future_dataframe(periods=12, freq='M')
    prophet_forecast = prophet_model.predict(prophet_forecast)
    
    prophet_model.plot(prophet_forecast,xlabel='',ylabel='sales'); plt.show()
    

    ## forecasting Sales Decomp.Trend
    decomp_trend_sales = pd.DataFrame(decomp_trend_) 
    decomp_trend_sales = decomp_trend_sales.set_index(total_sales.index)
    decomp_trend_sales = decomp_trend_sales.reset_index()
    decomp_trend_sales = decomp_trend_sales.rename(columns={decomp_trend_sales.columns[0]:'ds',decomp_trend_sales.columns[1]:'y'})
    decomp_trend_sales.head(1)

    prophet_model_sales = Prophet(changepoint_prior_scale=0.15,growth='linear',yearly_seasonality=True,seasonality_mode='multiplicative')
    prophet_model_sales.fit(decomp_trend_sales)
    
    prophet_forecast_sales = prophet_model_sales.make_future_dataframe(periods=12, freq='M')
    prophet_forecast_sales = prophet_model_sales.predict(prophet_forecast_sales)
    
    prophet_model_sales.plot(prophet_forecast_sales,xlabel='',ylabel='log-sales'); plt.show()
    
    prophet_model_sales.changepoints
    
    print(prophet_model_sales)
    
    trend_fcst_sales = prophet_forecast_sales.yhat[-12:]
    
    ## forecasting Adjusted Sales Decomp.Trend
    decomp_trend_ = pd.DataFrame(decomp_trend) 
    decomp_trend_ = decomp_trend_.set_index(total_sales.index)
    decomp_trend_ = decomp_trend_.reset_index()
    decomp_trend_ = decomp_trend_.rename(columns={decomp_trend_.columns[0]:'ds',decomp_trend_.columns[1]:'y'})
    decomp_trend_.head(1)

    prophet_model = Prophet(changepoint_prior_scale=0.15,growth='linear',yearly_seasonality=True,seasonality_mode='multiplicative')
    prophet_model.fit(decomp_trend_)
    
    prophet_forecast = prophet_model.make_future_dataframe(periods=12, freq='M')
    prophet_forecast = prophet_model.predict(prophet_forecast)
    
    prophet_model.plot(prophet_forecast,xlabel='',ylabel='log-sales'); plt.show()
    
    prophet_model.changepoints
    
    print(prophet_model)
    
    trend_fcst = prophet_forecast.yhat[-12:]


# Net Sales (not adjusted)
trend_fcst_ = trend_fcst_sales.reset_index(drop=True); len(trend_fcst_sales)
seasonal_fcst_ = seasonal_fcst_.reset_index(drop=True); len(seasonal_fcst_)
resid_fcst_.shape

# Net Sales Adjusted
trend_fcst = trend_fcst.reset_index(drop=True); len(trend_fcst)
seasonal_fcst = seasonal_fcst.reset_index(drop=True); len(seasonal_fcst)
resid_fcst.shape

# MC simulation for Net Sales Trend Forecast
fcst_ = np.ndarray(shape=(12,resid_fcst_.shape[1]), dtype=float)
for j in range(0,resid_fcst_.shape[1]):
    fcst_[:,j] = trend_fcst_ + seasonal_fcst_ + resid_fcst_[:,j]
fcst_mean_ = fcst_.mean(axis=1)

# MC simulation for Net Sales Adjusted Trend Forecast
fcst = np.ndarray(shape=(12,resid_fcst.shape[1]), dtype=float)
for i in range(0,resid_fcst.shape[1]):
    fcst[:,i] = trend_fcst + seasonal_fcst + resid_fcst[:,i]
fcst_mean = fcst.mean(axis=1)

# Forecast DataFrame
    fcst_date_index =  pd.date_range(start='01/01/2018', end='08/01/2022',freq='M') #prophet_forecast['ds']
fcst_plot = np.empty(shape=(len(total_sales_adj)+len(fcst_mean),1))
fcst_plot[-12:] = fcst_mean.reshape(-1,1)

fcst_plot_df = pd.DataFrame(np.exp(fcst_plot),index=fcst_date_index,columns=['fcst_adj'])

fcst_plot_df.loc[-12:,'fcst'] = np.exp(fcst_mean_) #.reshape(-1,1)

fcst_plot_df.loc[:len(total_sales_adj),'net_sales_adj'] = total_sales_adj.reset_index()['net_sales_adj'].values #fcst_plot_df = fcst_plot_df.join(.set_index(fcst_date_index))
fcst_plot_df.loc[:len(total_sales),'net_sales'] = total_sales.reset_index()['net_sales'].values 

    fcst_plot_df.iloc[len(total_sales_adj)-1, 0] = fcst_plot_df.iloc[ len(total_sales_adj)-1 , 2] # for plot to connect actuals to forecast
    fcst_plot_df.iloc[len(total_sales_adj)-1, 1] = fcst_plot_df.iloc[ len(total_sales_adj)-1 , 3] # for plot to connect actuals to forecast
 
fcst_plot_df = fcst_plot_df.round(1)

#fcst_plot_df.loc[:len(total_sales),'net_sales'] = total_sales.reset_index()['net_sales_adj'].values #fcst_plot_df = fcst_plot_df.join(.set_index(fcst_date_index))

fcst_plot_df.plot(title='Net Sales (Actual and Adjusted) and Forecasts (Optimist vs Conservative)'); plt.legend(loc='upper left'); plt.ylabel('$ Millions')





#### Correlation ####
cor_products = grouped_net_sales_pivoted.corr(method='pearson');
#cor = np.matrix(cor_products).flatten()
cor_products_list = pd.melt(cor_products)
    a = list(it.repeat(cor_products.index.to_list(),cor_products.shape[0]))
    b = list(it.chain(*a))
cor_products_list.index = b
cor_products_list_ranked = cor_products_list.sort_values(by='value',ascending=False)
    cor_products_list_ranked = cor_products_list_ranked[cor_products_list_ranked['value'] < .99999 ]
    cor_products_list_ranked = cor_products_list_ranked[cor_products_list_ranked['value'] > -.99999 ]
    cor_products_list_ranked = cor_products_list_ranked[cor_products_list_ranked['value'] != 'NaN' ]
    cor_products_list_ranked = cor_products_list_ranked[~cor_products_list_ranked['value'].between(-.6999,.6999) ]
    cor_products_list_ranked['value'].unique()
    cor_products_list_ranked = cor_products_list_ranked.reset_index()
    cor_products_list_ranked = cor_products_list_ranked.drop_duplicates('value')
    cor_products_list_ranked.columns = ['product_A','product_B','corr']
    cor_products_list_ranked.reset_index(drop=True)
print(cor_products_list_ranked)    


    
### Matrix of Monthly Change
total_sales_pct_change = total_sales.pct_change(1); print(total_sales_pct_change)
sales_chg_matrix = pd.DataFrame(index=list(range(1,13)),columns=list(range(2018,2022)))
    sales_chg_matrix.loc[:,2018] = total_sales_pct_change.values.round(3)[:12]
    sales_chg_matrix.loc[:,2019] = total_sales_pct_change.values.round(3)[12:24]
    sales_chg_matrix.loc[:,2020] = total_sales_pct_change.values.round(3)[24:36]
    sales_chg_matrix.loc[:7,2021] = total_sales_pct_change.values.round(3)[36:]
sales_chg_matrix['avg'] = sales_chg_matrix.mean(axis=1)
    sns.heatmap(sales_chg_matrix[[2018,2019,2020,2021]]) 

### Matrix of Yearly Change
total_sales_pct_change_YoY = total_sales.pct_change(12); print(total_sales_pct_change)
sales_chg_matrix_YoY = pd.DataFrame(index=list(range(1,13)),columns=list(range(2018,2022)))
    sales_chg_matrix_YoY.loc[:,2018] = total_sales_pct_change.values.round(3)[:12]
    sales_chg_matrix_YoY.loc[:,2019] = total_sales_pct_change.values.round(3)[12:24]
    sales_chg_matrix_YoY.loc[:,2020] = total_sales_pct_change.values.round(3)[24:36]
    sales_chg_matrix_YoY.loc[:7,2021] = total_sales_pct_change.values.round(3)[36:]
sales_chg_matrix_YoY['avg'] = sales_chg_matrix_YoY.mean(axis=1)
    sns.heatmap(sales_chg_matrix_YoY[[2018,2019,2020,2021]])

## Structural break
import statmodels.api as sm





## GROSS PROFIT : How much of each product has been profited from
grouped_gross_profit = pd.DataFrame(sale_type.groupby(['month','product_type'],sort=True)['gross_profit'].sum()).reset_index()
grouped_gross_profit = grouped_gross_profit.set_index('product_type')

grouped_gross_profit.groupby(['month'])['gross_profit'].nlargest(1) # Top Product Profits by Month

top_gross_profit = pd.DataFrame(grouped_gross_profit.groupby(['month'])['gross_profit'].nlargest(3)).reset_index()
top_gross_profit.set_index('month') # Top 3 products profits by month
top_gross_profit['product_type'].value_counts(sort=True) # summary of top products profits
top_gross_profit.groupby('product_type').sum().sort_values('gross_profit',ascending=False)


## GROSS MARGIN : How much of each product has been profited from percentage wise
grouped_gross_margin = pd.DataFrame(sale_type.groupby(['month','product_type'],sort=True)['gross_margin'].mean()).reset_index()
grouped_gross_margin = grouped_gross_margin.set_index('product_type')

grouped_gross_margin.groupby(['month'])['gross_margin'].nlargest(1) # Top Product Margins by Month

top_gross_margin = pd.DataFrame(grouped_gross_margin.groupby(['month'])['gross_margin'].nlargest(3)).reset_index()
top_gross_margin.set_index('month') # Top 3 products by month
top_gross_margin['product_type'].value_counts(sort=True) # summary of top products margins
top_gross_margin.groupby('product_type').mean().sort_values('gross_margin',ascending=False)


## REGION : How much of each region has been profited from net sales perspective
grouped_billing_region = pd.DataFrame(sale_type.groupby(['month','billing_region'],sort=True)['net_sales'].sum()).reset_index()
grouped_billing_region = grouped_billing_region.set_index('billing_region')

grouped_billing_region.groupby(['month'])['net_sales'].nlargest(1) # Top Pxroduct Margins by Month

top_billing_region = pd.DataFrame(grouped_billing_region.groupby(['month'])['net_sales'].nlargest(3)).reset_index()
top_billing_region.set_index('month') # Top 3 products by month
top_billing_region['billing_region'].value_counts(sort=True) # summary of top products margins
top_billing_region.groupby('billing_region').sum().sort_values('net_sales',ascending=False)


## COUNTRY (not US) : How much of each country has been profited from net sales perspective
(sale_type[sale_type['billing_country']!='United States']).head(3)
grouped_billing_country = pd.DataFrame(sale_type[sale_type['billing_country']!='United States'].groupby(['month','billing_country'],sort=True)['net_sales'].sum()).reset_index()
grouped_billing_country = grouped_billing_country.set_index('billing_country')

grouped_billing_country.groupby(['month'])['net_sales'].nlargest(1) # Top Product Margins by Month

top_billing_country = pd.DataFrame(grouped_billing_country.groupby(['month'])['net_sales'].nlargest(3)).reset_index()
top_billing_country.set_index('month') # Top 3 products by month
top_billing_country['billing_country'].value_counts(sort=True) # summary of top products margins
top_billing_country.groupby('billing_country').sum().sort_values('net_sales',ascending=False)

## Top selling product by top-seller State
df = sale_type[sale_type['billing_region'].isin(['California','Texas','Florida'])].reset_index(drop=True); print(df)
df_pivoted = df.pivot_table(values='net_sales',index='billing_region',columns='product_type',aggfunc=np.sum); print(df_pivoted)

for i in range(len(df_pivoted.index)):
    print(df_pivoted.iloc[i,:].sort_values(ascending=False)[:3])
    print('\n')


## Top selling product by top-seller foreign country
df = sale_type[sale_type['billing_country'].isin(['Canada','Germany','France'])].reset_index(drop=True); print(df)
df_pivoted = df.pivot_table(values='net_sales',index='billing_country',columns='product_type',aggfunc=np.sum); print(df_pivoted)

for i in range(len(df_pivoted.index)):
    print(df_pivoted.iloc[i,:].sort_values(ascending=False)[:3])
    print('\n')



## Correlations
cor = sales_plot.corr(method='pearson'); print(cor)
sns.heatmap(cor, cmap='Blues',vmin=-1, vmax=1)


## Histogram of Number of Orders
for product in ['Neck Gaiter','Mask With Nose Shape','Visor']:
    fig, ax = plt.subplots()
    sns.distplot(sales[sales['product_type'].isin([product])]['units_per_transaction'],ax=ax)
    ax.set(title=product)
    
for prod in ['Neck Gaiter','Mask With Nose Shape','Visor']:
    _ = plt.hist(sales[sales['product_type'].isin([prod])]['units_per_transaction'],color='red')
    _ = plt.legend([prod])
    plt.show()
    
