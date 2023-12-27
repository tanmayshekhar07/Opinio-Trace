import streamlit as st
import re
import subprocess
import boto3
import os
import pandas as pd
import time
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
from botocore.exceptions import NoCredentialsError, ClientError
import base64
from transformers import pipeline
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

# Global initialization of the classifier
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


# Create a session state for login status
if 'logged_in' not in st.session_state:
  st.session_state.logged_in = False
 
 
# Function to handle login
def handle_login(username, password):
    # Your login logic here
    if username == 'admin' and password == 'sentiment':
        st.success('Logged in successfully!')
        st.session_state.logged_in = True
        st.rerun()
    else:
        st.error('Wrong credentials')
 
# Login UI
def login():
    st.title("Login")
    col1, col2, col3 = st.columns([1,2,1])
 
    with col2:
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", key="password", type="password")
        
        if st.button('Login'):
            handle_login(username, password)
 
 
def download_and_read_file_from_s3():
    try:
        # AWS S3 Client Setup
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),  # Only if you're using a custom endpoint like MinIO
        )
 
        # S3 Bucket and Object Key
        s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        s3_object_key = 'reviews.csv'
 
        # Local file paths
        local_download_path = "/Users/riyavirani/NEU/Gen AI/amazon-python-scrapy-scraper-master/amazon/spiders/downloaded_file.csv"
 
        # Download file from S3
        s3_client.download_file(s3_bucket_name, s3_object_key, local_download_path)
        st.success("File downloaded successfully from S3.")
        st.write("Displaying data read from s3")
        df1 = pd.read_csv(local_download_path)
        st.dataframe(df1)

        # Read the synthetic data file using Pandas
        df = pd.read_csv("/Users/riyavirani/NEU/Gen AI/amazon-python-scrapy-scraper-master/amazon/spiders/synthetic_data.csv")
        st.success("File read into DataFrame.")
        st.write("Displaying synthetic data file")
        st.dataframe(df)
   
      
 
        # Format "date" as MM/DD/YYYY
        df['date'] = pd.to_datetime(df['date'])

        # Reformat to MM/DD/YYYY
        df['date'] = df['date'].dt.strftime('%m/%d/%Y')

    
 
        # Replace missing values in 'text' and 'title'
        df['text'].fillna('Not Available', inplace=True)
        df['title'].fillna('No Title', inplace=True)
 
        # Drop rows where 'rating' is missing
        df.dropna(subset=['rating'], inplace=True)
 
        # Combine 'title' and 'text' into a new column 'combined_text'
        df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
        df.drop(['title', 'text'], axis=1, inplace=True)
 
        # Text cleaning directly applied to the combined_text column
        df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'<.*?>', '', x))
        df['combined_text'] = df['combined_text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
        df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'@\w+|#', '', x))
        df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        st.write('Cleaned Data')
        df.to_csv('after_cleaning.csv')
        st.dataframe(df)
        
        # Applying sentiment analysis to each row
        df['sentiment'] = df['combined_text'].apply(sentiment_analysis)
        st.write("Sentiment Analyzed data")
        df.to_csv('sentiment_analyzed_file.csv')
        
        st.dataframe(df)

        plot_bar_graph(df)
        plot_avg_rating_over_time(df)
        sentiment_forecasting(df)
        
        
 
    except NoCredentialsError:
        st.error("Credentials not available for AWS S3.")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            st.error("The file does not exist.")
        else:
            st.error(f"Unexpected error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
 

def sentiment_forecasting(df):
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)
    
    # Map sentiment to numerical values: POSITIVE to 1, NEGATIVE to 0
    df['sentiment_score'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
    
    # Aggregate sentiment scores by date for the time series
    # Fill forward for any missing dates
    time_series = df['sentiment_score'].resample('D').mean().fillna(method='ffill')
    
    # Fit the ARIMA model
    # For simplicity, we'll start with order (1,0,1), but this should be fine-tuned
    model = ARIMA(time_series, order=(1, 0, 1))
    results = model.fit()
    
    # Forecast
    forecast = results.get_forecast(steps=30)  # Forecast the next 30 days
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(time_series.index, time_series, label='Observed')
    plt.plot(mean_forecast.index, mean_forecast, label='Forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_avg_rating_over_time(df):
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])



    # Group the data by 6-month periods and calculate the average rating
    df = df.set_index('date').resample('6M')['rating'].mean().reset_index()
    # Format the 'date' column to show only month and year
    df['date'] = df['date'].dt.strftime('%b %Y')
    # Plot
    plt.figure(figsize=(14, 7))
    barplot = sns.barplot(data=df, x='date', y='rating', errorbar=None, palette='viridis')

    # KDE plot
    kde_ax = plt.twinx()
    sns.kdeplot(data=df, x='rating', ax=kde_ax, color='crimson', label='KDE')

    # Enhancing the plot to look like the second picture
    kde_ax.yaxis.label.set_color('crimson')
    barplot.set_xlabel('Date (6-month periods)', fontsize=14)
    barplot.set_ylabel('Average Rating', fontsize=14)
    kde_ax.set_ylabel('Density', fontsize=14, color='crimson')
    plt.title('Average Rating Over Time', fontsize=16)

    # Tilt the x-axis labels
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)

    # Show the legend
    plt.legend(loc='upper left')

    # Adjust the tick parameters
    barplot.tick_params(axis='y', colors='blue')
    kde_ax.tick_params(axis='y', colors='crimson')

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()





 

 
# Function to filter data by rating
def filter_data_by_rating(df, rating):
    if rating == "All Ratings":
        return df
    return df[df['rating'] == int(rating)]
 
# Function to plot the histogram of rating counts per month
def plot_rating_histogram(df, selected_rating):
    # Prepare the dataframe for plotting
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    if selected_rating != "All Ratings":
        df = df[df['rating'] == selected_rating]
    rating_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
 
    # Ensure all months are shown
    all_years = pd.DataFrame({'year': df['year'].unique()})
    all_months = pd.DataFrame({'month': range(1, 13)})
    rating_counts = pd.merge(all_years, rating_counts, on='year', how='left')
    rating_counts = pd.merge(all_months, rating_counts, on='month', how='left').fillna(0)
 
    # Pivot the data for easy plotting with years on x-axis and months as hue
    rating_counts_pivot = rating_counts.pivot('year', 'month', 'count').fillna(0)
 
    # Plot
    plt.figure(figsize=(15, 6))
    sns.barplot(data=rating_counts, x='year', y='count', hue='month', palette='viridis', ci=None)
    plt.title(f"Yearly Count of {'All Ratings' if selected_rating == 'All Ratings' else 'Rating ' + str(selected_rating)}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Count of Ratings", fontsize=14)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
 










 
def sentiment_analysis(text):
    try:
        truncated_text = text[:512]
        result = classifier(truncated_text)
        return result[0]['label']
    except Exception as e:
        return f"Error during sentiment analysis: {e}"



def plot_bar_graph(df1):
    try:
        # Count the number of positive and negative reviews
        sentiment_counts = df1['sentiment'].value_counts()

        # Create a bar chart showing the number of positive and negative reviews
        ax = sentiment_counts.plot(kind='bar', color=['blue', 'red'])
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.title('Number of Positive and Negative Reviews')

        # Annotate bars with data labels
        for bar in ax.patches:
            # Get the height and position of each bar
            y_value = bar.get_height()
            x_value = bar.get_x() + bar.get_width() / 2

            # Format and place the data label
            label = format(y_value, ',')  # format the count with a comma as a thousands separator
            plt.annotate(
                label,                      # use `label` as label
                (x_value, y_value),         # place label at end of the bar
                textcoords="offset points", # specify the position of the label
                xytext=(0,5),               # slight offset from the top of the bar
                ha='center')                # align the label horizontally to center

        # Use Streamlit's pyplot function to display the plot
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in plotting bar graph: {e}")






def trigger_airflow_dag(product_code):
    airflow_url = 'http://localhost:8080/api/v1/dags/run_scrapy_spider_with_product_code/dagRuns'
    airflow_username = os.environ.get('AIRFLOW_USERNAME')
    airflow_password = os.environ.get('AIRFLOW_PASSWORD')
 
    json_payload = {
        "conf": {
            "product_code": product_code
        }
    }
 
    response = requests.post(airflow_url, json=json_payload, auth=HTTPBasicAuth(airflow_username, airflow_password))
 
    if response.status_code in [200, 201]:
        st.success("DAG triggered successfully!")
        time.sleep(60)
        # Call the function
        download_and_read_file_from_s3()
        
    else:
        st.error(f"Failed to trigger DAG: {response.text}")
 
 
 
# Main application
def main():
    if not st.session_state.logged_in:
        login()
    else:
        st.title("Product Review Analysis")
        url_input = st.text_input('Enter Amazon product URL:')
        if url_input:
            if st.button('Search'):
                pattern = r'dp/(.+)/ref|\/dp\/([^\/]+)'
                match = re.search(pattern, url_input)
                if match:
                    product_code = match.group(1) or match.group(2)
                    st.write(f"Extracted product code: {product_code}")
                    try:
                        trigger_airflow_dag(product_code)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error('Unable to extract product code from the provided URL.')
 
 
 
if __name__ == '__main__':
    main()