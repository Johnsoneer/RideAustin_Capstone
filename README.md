# Capstone: RideAustin Hotspots

My Capstone project in which I use the RideAustin ride-request database to predict where drivers should place themselves to maximize ridership and minimize unfulfilled requests.


# Update: 6/5/17
This project is currently ongoing. I am currently using a K-Means Clustering algorithm to determine where the 5 best locations are for drivers to situate themselves based on the previous two hours worth of ride-requests. Using Bokeh to plot these locations, I've discovered that while downtown seems to harbor the most requests, both completed and unfulfilled, the optimal locations outside of downtown tend to vary based on the time of day and day of the week. 

I plan on including a new feature for current-weather-outlook to more accurately predict the volume of requests in a given half-hour timeblock. My theory is that rain will cause more riders to ask for rides across the map. 
