import pandas as pd
import numpy as np
import mysql.connector as msc
import datetime as dt
from collections import Counter
from bokeh.io import output_file, show, output_notebook
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
from sklearn.cluster import KMeans
import random
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import vincenty
from sklearn.cross_validation import train_test_split, cross_val_score


class HotspotPredictor(object):

    def __init__(self,engine):
        #initialize dataframe
        self.df = pd.read_sql(
            'SELECT start_location_lat,start_location_long, created_date,tod \
            FROM rides  \
            WHERE status = "NO_AVAILABLE_DRIVER" \
            OR status = "Completed";', engine)
        #clean out data outside Austin, TX
        self.df['day_of_week'] = self.df['created_date'].dt.weekday
        self.df = self.df[(self.df['start_location_lat'] >= 30.190833) & (self.df['start_location_lat'] <= 30.404041)]
        self.df = self.df[(self.df['start_location_long'] >=-97.819014) & (self.df['start_location_long'] <= -97.647192)]
        self.df['period'] = df.apply(self.period, axis=1)



    def period(self,row):
        '''
        To be .apply()'d to self.df with 'created_date' column.
        This takes in a row and creates a new column that
        assigns that row to a particular 30 minute timeblock.
        '''
        timelables = list(range(0, 49))
        timevalues = []
        for x in list(range(0,25)):
            timevalues.append((x,0))
            timevalues.append((x,30))
        periods = dict(zip(timelables, timevalues))
        visit_start = {'hour': row.created_date.hour, 'min': row.created_date.minute} # get hour, min of visit start
        for label, tupe in periods.items():
            hour = tupe[0]
            thirty = tupe[1]
            if hour == visit_start['hour']:
                if thirty <= visit_start['min'] <= thirty+30:
                    return label
                else:
                    return label+1



        def create_centroids(self, dataframe):
        '''
        Takes a dataframe of my start_location_lats and start_location_longs and builds a K-Means model with 5 centroids.
        It returns a numpy array of the centroids (by lat-long pair) and a dictionary where the key is the centroid rank
        and the value is a list of the [lat,long,# of datapoints, rank] for that centroid.

        INPUT:
        - Dataframe
        OUTPU:
        - numpy array
        - dictionary'''
        if type(dataframe) == str:
            return dataframe
        X = np.array(dataframe[['start_location_lat','start_location_long']])
        model = KMeans(n_clusters=5)
        model.fit(X)
        cents = model.cluster_centers_
        lables_model = model.labels_
        c = Counter(lables_model)
        centroids_by_intensity = c.most_common(5)
        ordered_labels = [i for i,x in centroids_by_intensity]
        ordered_centroids = []
        centroid_dict = {}

        for i, index in enumerate(ordered_labels):
            ordered_centroids.append(cents[index])
            centroid_dict[i] = [cents[index][0],cents[index][1],centroids_by_intensity[i][1],i]

        return np.array(ordered_centroids), centroid_dict




    def plot_hotspots(self,centroids, centroid_dictionary,num_datapoints, completed_rides=None, unfulfilled_rides=None):
        '''
        Takes in centroid values from self.create_centroids() and centroid_dictionary and plots the centroids relative to their
        intensity. Optional inputs for the lat-long columns for completed_rides (green) and unfulfilled_rides(blue).

        INPUT:
        - centroids (numpy array)
        - centroid_dict (dictionary)
        - copmleted_rides (dataframe)(optional)
        - unfulfilled_rides (dataframe)(optional)

        OUTPUT:
        -None
        '''
        #creating the plot
        map_options = GMapOptions(lat=30.29, lng=-97.73, map_type="roadmap", zoom=11)
        plot = GMapPlot(
            x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
        )
        plot.title.text = "Austin"
        plot.api_key = "AIzaSyBx-cLXm4jxpg0aX_nnUnwd2hir3Ve0j9w"

        #create alpha based on intensity
        alpha = []
        for key, value in centroid_dictionary.iteritems():
            al_value = value[2]/float(num_datapoints)
            al_fixed = al_value+.25
            alpha.insert(key,al_fixed)

        #try if completed_rides is populated
        try:
            completed_lats = list(completed_rides['start_location_lat'])
            completed_longs = list(completed_rides['start_location_long'])
            completed_source = ColumnDataSource( data=dict(
                lat=completed_lats,
                lon=completed_longs,))
            completed_dots = Circle(x="lon", y="lat", size=15, fill_color="green", fill_alpha=0.1, line_color=None)
            plot.add_glyph(completed_source, completed_dots)
        except:
            pass

        #try if unfulfilled_rides is populated
        try:
            unfulfilled_lats = list(unfulfilled_rides['start_location_lat'])
            unfulfilled_longs = list(unfulfilled_rides['start_location_long'])
            unfulfilled_source = ColumnDataSource(
            data=dict(
                lat=unfulfilled_lats,
                lon=unfulfilled_longs,))
            unfulfilled_dots = Circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
            plot.add_glyph(unfulfilled_source, unfulfilled_dots)
        except:
            pass
        #creating centroid source and circle
        centroidlats = centroids[:,0]
        centroidlongs = centroids[:,1]
        centroid_source = ColumnDataSource(
            data=dict(
                lat=centroidlats,
                lon=centroidlongs,
                 alpha=alpha))
        centroid_dots = Circle(x="lon", y="lat", size=45, fill_color='#8B008B', fill_alpha='alpha', line_color=None)
        plot.add_glyph(centroid_source, centroid_dots)

        #finishing the plot
        plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
        show(plot)



    def find_similar_pairs(self,dataframe,ride_request):
        '''
        Takes in a dataframe to test against and a ride request. Populates a distribution of other rides that
        could follow that particular ride.


        input:
        -original dataframe.values (ndarray)
        -row (ndarray)
        output:
        -list of possible new points to sample from (Dataframe)
        '''

        following_rides_list = []
        row1 = ride_request
        for i, request in enumerate(dataframe):
            if row1[5] <= request[5] <= row1[5]+1 and row1[4] == request[4] \
            and row1[0] -.004 <= request[0] <= row1[0]+.004 \
            and row1[1]-.004 <= request[1] <= row1[1]+.004:
                try:
                    request2 = dataframe[i+1]
                except:
                    continue
                following_rides_list.append(request2)
        return pd.DataFrame(following_rides_list, columns=['start_location_lat',\
        'start_location_long','created_date','tod','day_of_week','period'])




    def predict_rides(self, input_dataframe):
        '''
        Takes in the current dataframe and predicts the next half-hour worth of ride quests using
        self.find_similar_pairs() method.

        Inputs:
        dataframe,
        row,
        n_rides (to be predicted in separate function)

        output:
        Dataframe (predicted ride requests)'''

        predicted= []
        last_ride = input_dataframe.tail(1)
        datetime = last_ride['created_date']
        last_ride = last_ride.values[0]
        values = input_dataframe.values
        n_rides = self.predict_n_rides(input_dataframe,datetime)
        if n_rides < 20:
            return "not enough data to predict hotspots"
        for rep in xrange(n_rides):
            distribution = self.find_similar_pairs(values,last_ride)
            sample = distribution.sample().values
            if len(sample) < 5:
                sample = sample[0]
            predicted.append(sample)
            last_ride=sample
        return pd.DataFrame(predicted, columns=['start_location_lat','start_location_long','created_date',\
                                                'tod','day_of_week','period'])


    def distance_error(self,predicted_centroids,actual_centroids):
        '''
        takes in two sets of centroids and returns the average distance between each predicted centroid and the
        closest actual centroid

        input:
        -predicted ndarray
        -actual ndarray

        output:
        -float
        '''

        distances = []
        for cent in predicted_centroids:
            closest = 10000
            for i in actual_centroids:
                if vincenty(cent,i).miles < closest:
                    closest = vincenty(cent,i).miles
            distances.append(closest)
        return np.mean(distances)



    def prep_df(self,data):
        '''
        takes in dataframe of raw data and cleans it so that it can be passed
        into my regression model to predict n_rides for the next half-hour block.

        input:
        -dataframe

        output:
        -X dataframe
        -y dataframe
        '''

        data['month']= data['created_date'].dt.month
        data['day'] = data['created_date'].dt.day
        data['day_of_week'] = data['created_date'].dt.weekday
        # creating a COUNT response variable
        data = data.groupby(['day_of_week','period','day'])['start_location_lat'].count().reset_index(name="count")
        #data = pd.get_dummies(data, columns=['day_of_week'], drop_first = True)
        y = data.pop('count')
        X = data
        return X,y


    def predict_n_rides(self,data,datetime):
        '''
        takes in data and a split datetime and predicts how many n_rides
        there will be for the next half hour.

        input:
        - data (dataframe)
        - datime (string or datetime obj)

        output:
        -int
        '''

        data,response = prep_df(data)
        rf1 = RandomForestRegressor(n_estimators=100)
        model = rf1.fit(data,response)
        data2, response2 = prep_df(d2)
        n_rides = model.predict(data2).astype(int)[0] #return prediction as integer
        if n_rides <= 10:
            print "Error: Not enough ride-traffic to predict hotspots."
        return n_rides



    def random_date():
        '''
        a random datetime generator to use for randomly splitting my dataset
        in the create_random_tests() method.
        '''
        year = 2016
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(0,23)
        minute = random.randrange(0,31,30)
        return '2016-%s-%s %s:%s:00' %(month,day,hour,minute)



    def create_random_tests(self,df,n_repititions=100):
        '''
        testing my model to find where the mean distance-error is. Use this distribution to create a hypo test to find
        where the true mean is.

        input:
        none

        output:
        list of floats.
        '''

        errors = []
        for x in xrange(n_repititions):
            #define parameters:
            date = self.random_date()
            dated = pd.to_datetime(date)
            before,after = self.split_at_date(df,dated)
            if len(after) <= 5:
                continue
            print 'date working: %s' %(date)

            # test_centroids:
            test_centroids = self.create_centroids(after)
            print 'test_centroids_working'

            #train_centroids:
            try:
                predictions = self.predict_rides(before)
            except:
                pass

            train_centroids = self.create_centroids(predictions)
            if type(train_centroids) == str:
                continue
            print'train_centroids_working with %s predicted rides' %(len(predictions))

            #find error:
            error = self.distance_error(train_centroids[0], test_centroids[0])
            errors.append(error)
            print 'completed error! %s' %(error)
        return errors
