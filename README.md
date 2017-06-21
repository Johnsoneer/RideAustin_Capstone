
# Capstone: RideAustin Hotspots

![app_gif_4_bigger](https://user-images.githubusercontent.com/24977834/27398106-5409cdc6-567e-11e7-886b-df224e8d6e4e.gif)



My Capstone project in which I use the RideAustin ride-request database to predict where drivers should place themselves to maximize ridership and minimize unfulfilled requests.

# The story

On April 13th of 2013, I went to a concert. Specifically here, in Downtown Austin Texas.

![concert_spot](https://user-images.githubusercontent.com/24977834/27397925-9a48f362-567d-11e7-9b96-76da625680ee.png)

After the show was done at 10:00pm, I needed a ride home. So I opened up my favorite ride-sharing app and asked for a ride. But instead of getting a ride, I got a message that read "NO DRIVER AVAILABLE". I thought that was odd, since this was the middle of downtown and there should be plenty of traffic. So I gave it a minute and sure enough, after trying again I got myself a ride.

I told the driver, "Boy, you guys must be busy if I can't get a ride downtown."

He looked back at me and said "Actually, I've been driving around for ages looking for a fare."

What that told me was that there was a disconnect between rider and driver. Here I am with money, he's got a car, but the only thing keeping us from finding each other is location.

# My Goal

The purpose of my project is to try to bridge that gap. If I can successfully predict what areas in Austin are likely to have high demand, I can tell drivers about those spots and increase the supply to meet that demand. Predicting where these hotspots are going to be means taking in the historical information about when and where ride requests happen and using them to predict where the ride requests are likely to happen next.

# Choosing a Model


At first I considered breaking the whole of Austin into a grid and capturing information about each grid point to predict which one is going to have the highest demand. I read about some other similar projects doing this with some success in areas like Japan where a cell service company was able to predict ride requests within 500 square feet. What they had that I don't is real-time geolocation of users.

While the grid-mapping method may make it easier for a model like an ensemble classifier to predict the area with the highest demand, what it also does is decorrelate areas from each other. If I have Downtown blocked off into a grid of 4 areas, then a classifier would assume each class is independent of each other when I know that all four of those areas are highly correlated with each other. More over, I wanted to capture the variability of these areas of demand continuously to see how they may change in certain circumstances. So I instead decided to plot hotspots using a K-Means classifier.

# K-Means Clustering

![step_6_calculate_hotspots](https://user-images.githubusercontent.com/24977834/27398701-367099f0-5680-11e7-9c08-c1c9af710dd9.png)

This is an unsupervised machine-learning algorithm that separates a dataset into a certain number (k) of clusters. If I can find the center of those clusters, known as the centroid, then those could be good hotspots to suggest to my drivers. The trick is getting the right number of clusters down. I found that 5 clusters makes the most sense for Austin's ride request data, given the neighborhoods and historical ride demand.

These hotspots are returned as pairs of Latitude and Longitude coordinates. The trouble with running these kinds of pairs into a regressor is that if I were to predict a certain latitude, how do I know which longitude makes sense to pair it with? Moreover, regressors are designed to predict values with the highest accuracy to out-of sample data, but do I really care about what order my predictions are returned in? No. All I care about is that my predicted hotspots make sense with where the demand is actually going to be. I want to send my drivers to the right spots, not necessarily to the right spots in a particular order.

# My Model

In the end, I created an algorithm that would use historical data in order to predict likely ride-requests, then run a k-means clustering algorithm on those rides to find where the hotspots are going to be. Breaking it down into six steps, the model works like this:

# Step 1: Find a ride request

![step_1_pick_ride](https://user-images.githubusercontent.com/24977834/27398930-df690f74-5680-11e7-9f39-d491768f8439.png)

Let's look at just one ride. I've got information about it, like it's lat-long location, the time of day, the day of the week, and the date and so on. If I go back into my dataset, looking at all the ride-requests that came before this one, I can then find requests that are *similar* to that ride request.

# Step 2: Find Similar Requests

![step_2_find_similar](https://user-images.githubusercontent.com/24977834/27399031-25787b62-5681-11e7-85a2-e1a719cacad7.png)

The green dots are all similar rides that happened around the same spot, same time of day, and the same day of the week. Each of these rides had at least one ride request to come *after* it. My thought was that those following rides would follow a historical pattern of where rides were likely to prop up next in similar conditions. So let's find the requests that came after all these green ones.

# Step 3: Find the Following Rides

![step_3_find_potential_followups](https://user-images.githubusercontent.com/24977834/27399152-77ddf562-5681-11e7-8bc2-26dfef117fce.png)

These green dots are all rides that came after the similar ride requests to my original blue one, down in the bottom left corner. As you can see, most of them occur Downtown, the bar scene south of the river, the mall, or the airport. This being 10:00pm on a Thursday night, that makes sense.

So let's now randomly sample from this distribution of requests and choose that to be our next expected ride request. This is sort-of hijacking the Central Limit Theorem in statistics that tells us the more observations we take, the more normal our distribution should be and the more towards the true-mean our observed mean should be. In this case, we have most of our rides downtown, but a good amount elsewhere. An independent random sample of this distribution should reflect the same variation of the ride requests that would follow in reality.

# Step 4: Randomly sample

![step_4_random_sample](https://user-images.githubusercontent.com/24977834/27399256-c6834032-5681-11e7-8e79-1c4ee366fbfe.png)

See that new blue dot? That is our first *expected* request. It's highly unlikely that the next actual ride request is going to be in that exact spot, but that's now what we're after. We want to know what the general distribution of rides is likely to look like, so we can get a good idea of what areas might see more demand in the coming 30 minutes.

So we repeat the process, finding similar ride requests and their follow ups and plotting the distribution of those, we can see they again favor down town but also spread out a little.

# Step 5: Repeat

![step_5_repeat_n_times](https://user-images.githubusercontent.com/24977834/27399363-1f94006c-5682-11e7-91d7-691603aef182.png)

If we were to repeat this process enough times, we now have a distribution of what we expect our rides to look like for the next 30 minutes. How did we decide how many to pick? I used a regression model called a Random Forest to predict how many rides we should expect to see for the whole of Austin in the next 30 minutes. That number, we'll call *n*, is how many times we repeat this process. If we repeated the same amount every time, then our expected hotspot for any given time period would generally be the average hotspot for that time block. While that might be ok to do, I'm more interested in capturing the variation on any given day.

So now that we've got an idea of how we expect the demand to look in the next half our, let's run our K-Means clustering algorithm on it.

# Step 6: Cluster

![step_6_calculate_hotspots](https://user-images.githubusercontent.com/24977834/27399727-618db57a-5683-11e7-855b-384805e2d14f.png)
Now we've got our hotspots!

These are the areas we'll suggest to our riders to pay attention to. Now, in data science, a model is no good unless we can prove that it works. So let's test this method by calculating the distance between the *predicted* hotspots and the closest *actual* hotspots that we can generate using the data we've withheld from the model so far. We should come up with 5 distances in miles. We'll use the 'Manhattan Distance' since that is more representative of driving distance for our case. Take the average distance of those 5, and we'll get a good idea of how well our model predicted hotspots.

# How did we do?

Let's go back to April 13th to see how we did. That bigger purple dot is a more precise look at where my model suggested drivers go. The blue dots are ride requests that happened between 10 and 10:30 that night. That blue one in the top right corner? That's me!


![actual vs prediciton](https://user-images.githubusercontent.com/24977834/27400394-92171d2e-5685-11e7-93f9-edb5c4029653.png)


# Running Random Tests:

Now let's repeat this process of testing the distance in miles between our predicted hotspots and our actual hotspots. Below is the result of randomly testing my model at 120 different dates and seeing how well this process worked for each date.

Note: I split the dataset in half each time, only using the 'before' data to train the model so that there was no leakage of information.

Note2: There were often times when there was not enough predicted traffic to justify saying that there were any hotspots going to happen. If a Monday morning at 3am only expects to see six ride requests, then I can't really say that any location would be a 'hot' spot to be in.

![testing_the_mean_error](https://user-images.githubusercontent.com/24977834/27399985-5170c9e2-5684-11e7-8981-73e11d62cdbc.png)

Via hypothesis test, I can conclude that on average, my model predicts hotspots within 2 miles of the actual hotspot that occurred afterwards. While 2 miles is good, I believe I could get closer if I added in some more details about the rides.

# Where to go from here?

One bit of information I could use to improve this model is weather data, like % chance of percipitation or temperature. Anyone who has ever tried to get a cab in New York City while it's raining knows that demand skyrockets and all the cabs are taken. Likewise, I'd also like to know more about incoming flights at the airport. If I can predict a certain number of flights coming in at the same time, I may suggest the airport is a better spot than usual.

As I mentioned earlier, a company in Japan was able to use a mobile user's real-time geolocation data to predict when and where they would ask for a ride. Some taxi drivers saw increases of 20% in business after using the model. While I don't have that kind of data, if I did you can bet I would use it.

Something else I never got to try was weighting the unfulfilled rides more. If I care more about taking care of unfulfilled requests, then my hotspots should favor those.

# Limitations

As I noted before, there are certain time periods where the model does not work. Hard to say any spot could be 'hot' if there will only be a handful of ride requests happening in the whole of Austin.

Another downside may be that since I let the hotspots roam wherever they want, the more precise locations can sometime be in the river, or perhaps off road. The general area might still be good, but I need to be careful not to tell drivers to submerge their car in the lake.

Lastly, it's important to consider Herd-Behavoir here. If I tell my drivers that one location is great, and they all flock there, I may be draining the supply of drivers in other locations. This might have an adverse affect on my ability to meet overall demand and instead focus too hard on high demand areas. Testing this for real would tell me more about what to set my threshold at for this.

# App

So how might I present this information to Drivers? If you've ever looked at a weather radar, you know that in one setting it shows you the past hour's worth of the storm as it marches across your screen. Then, if you hit 'future', it will then show you where the model expects the storm to go.

I've organized a dashboard using Tableau to do the same. Here I have two gifs demonstrating looking at the past hour's worth of data and then presenting my expectations of where the demand is likely to be next.

![app_gif_1_bigger](https://user-images.githubusercontent.com/24977834/27400689-7c301bae-5686-11e7-9a7c-78c204cd7bf6.gif)

Expected:
![app_gif_2_bigger](https://user-images.githubusercontent.com/24977834/27400717-938c4cbe-5686-11e7-88f8-7a28e3cc8adc.gif)


That may work well for any city, but what if I know things specifically about areas in Austin that I want to take advantage of? Below you'll see how I provide the same previous vs. expectation format to specific areas like Downtown, the Airport, or the Northern part of South Lamar street.

![app_gif_3_bigger](https://user-images.githubusercontent.com/24977834/27400754-bfcda9b2-5686-11e7-97f2-dda9d2b744ab.gif)

Expected:
![app_gif_4_bigger](https://user-images.githubusercontent.com/24977834/27400787-d38aa860-5686-11e7-8486-6b34c2579bee.gif)

Notice how the northern part of South Lamar street has a good amount of demand, nearly $20 on average for fares (which is a good amount!) but only 30% of the requests are actually getting taken! What that tells me, the driver, that I could get a good amount of money without having to wait very long because there is demand there that is not getting met.

Double checking this, we look at the expected demand for that area, and we can see that it is expected to go up and we can see anywhere between 0 to 28 requests to happen in the next half hour (that's a 95% confidence interval by the way). So now I know that this is a good area to be in so I'm going to sink my gas money into going there instead of waiting at the airport any longer for a bigger fair but fighting with all the other drivers for the next ride.

# Wrapping Up

A huge thank you to the folks at Ride Austin for allowing me to use their dataset for this project. Also, big thanks to Tableau for giving me a student license to learn how to use their software to build this awesome interactive app you see above.

If I could, I would go into greater detail in the ways I mentioned earlier with this project. Providing the drivers with this kind of information allows THEM to be the ones making decisions on where to go. They want to give people rides! What my model does is help them find where they are and where they might be next.
