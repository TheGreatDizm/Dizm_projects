# Our data set illustrates 100 customers in a shop, and their shopping habits.

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

# Result:
# The x axis represents the number of minutes before making a purchase.
# The y axis represents the amount of money spent on the purchase.

# Split Into Train/Test
# The training set should be a random selection of 80% of the original data.
# The testing set should be the remaining 20%.

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Display the Training Set
# Display the same scatter plot with the training set:

# Display the Testing Set
# To make sure the testing set is not completely different, we will take a look at the testing set as well.

mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))

myline = np.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# The result can back my suggestion of the data set fitting a polynomial regression, even though it would give us some weird results if we try to predict values outside of the data set. Example: the line indicates that a customer spending 6 minutes in the shop would make a purchase worth 200. That is probably a sign of overfitting.
# But what about the R-squared score? The R-squared score is a good indicator of how well my data set is fitting the model.

from sklearn.metrics import r2_score

r2_1 = r2_score(train_y, mymodel(train_x))

print(r2_1)

# Note: The result 0.799 shows that there is a OK relationship.

# Bring in the Testing Set
# Now we have made a model that is OK, at least when it comes to training data.
# Now we want to test the model with the testing data as well, to see if gives us the same result.

r2_2 = r2_score(test_y, mymodel(test_x))

print(r2_2)

# Note: The result 0.809 shows that the model fits the testing set as well, and we are confident that we can use the model to predict future values.

# How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?

print(mymodel(5))
