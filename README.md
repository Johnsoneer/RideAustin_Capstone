# Capstone: RideAustin Hotspots

My Capstone project in which I use the RideAustin ride-request database to predict where drivers should place themselves to maximize ridership and minimize unfulfilled requests.


# Update: 6/12/17
This project is currently ongoing. I've been able to build a model based on both supervised and unsupervised algorithms that predict hotspots within Austin city limits for the next half hour. My testing metric is the mean-distance-error from my predicted hotspots to the actual hotspots observed in my test data, measured in miles. 

Via a classical hypothesis test, I can say with 99.99% confidence that my model's preditions are on average within 2 miles of the actual hotspot. 

Next step is taking this information and compiling it in a dashboard that drivers can use to easily interact with this kind of information. Thinking about using Tableau for this.
