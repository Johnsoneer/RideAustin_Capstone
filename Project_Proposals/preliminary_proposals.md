Preliminary_Proposals: Will Johnson

#1:Ride-Austin
I know I have data for this one, it's available on Dataworld.net. The basic idea here is to create a model for the drivers of Ride Austin to maximize their earnings. Take the 2016 rides dataset and try to predict where a driver should position themselves to make the most money for their next ride based on historical data.

The question I'm trying to solve is one drivers of ride-sharing services face: Where do I go to maximize my earnings? Some choose the airport because they're frequent arrivals with tons of passengers. Others choose downtown because of surge pricing. I'd like to solve this problem with data.

My presentation on this particular one could be one of two things. I know I can use a visualization map to show how certain drivers might make more money in certain areas, but what I'd like to do (I'd need some practice in web-app development to do it) is to create an actual web app that will display a realtime map that will ask for your driver's information and spit out expected values of your next fare.

As I mentioned, the data is already available on Dataworld.net, I can have it on my computer in a matter of minutes.

My next step here, aside from actually getting the data, would be deciding what kind of model am I going for here? Just a thought, maybe I would want to create my own grid-search for a regressor that returns the expected value of my returns along with what zip-code or lat-long location I need to be at to achieve that return.

#2: Home-away vs. AirBnB
This one, the data is a little more questionable but something I could probably put together. Like the first idea, this model would be predictive for the renters (not rentees) of a vacation-rentals service. The model I build would be able to suggest which service I should use for renting my apartment to maximize my expected returns.

My presentation, again like the first idea, would likely be slides and perhaps some visualization engine to help me demonstrate where certain homes do better, but if I could I'd like to build a web-app that can actually take in the features of a home and return the expected returns on both AirBnB and on Home-Away for my situation.

AirBnB has this data available online already for anyone who wants it, which is nice, but Home-Away is a little harder to get at. Ideally I might be allowed the data from Home-Away themselves, but if that fails they do have an API I can scrape from. The problem then would be making sure the data I am able to scrape is as valid and as useful as the AirBnB dataset.

Next step would be beginning to scrape the Home-Away api and trying to find what features I can use for both services.

#3: Bands on the Run
This one will be VERY difficult to get the data for, but if a label is willing to play ball, it might work.

I'd like to create a predictive model that will allow bands to know what cities are going to maximize their return while on tour. Indie bands often only have enough resources to venture out to one or two cities outside their hometown, so which one to go to makes a big impact.

My presentation would probably demonstrate the model with a visualization engine that would allow me to demonstrate how certain bands have faired better in certain cities. My data sources could be difficult here. While I COULD go to spotify/social media sites to scrape information about indie bands both before and after a tour date, I would argue that social media boosts are not the whole picture when it comes to deciding which city to go to on tour. Bands should be concerned about conversions (i.e. making someone go from discovery to dedicated fan willing to pay for something). Some cities have explosions of social media for an indie band, but those social media bombs often fade and result in few conversions because of how much competition there is in that city and how much social media activity there is.

For that reason, I would want more, like their merch sales, download numbers, and some metric on business contacts if possible. (i.e. label staffer who reached out after the show). To get my hands on that kind of data, it would need to be from a label (who actually has it).

Next steps, beyond getting the data, would be trying to build a metric to judge how valuable a tour date at a certain city is to a band. It needs to combine things like merchandise sales, ticket sales, social media numbers, and streaming figures and how much they all increased after the concert. 
