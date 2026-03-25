'''
SPAM E-MAIL FILTERING MACHINE

Created by ADITYA GANGULY ON 25.03.2026

'''

#importing required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

# in this example , we create a small dataset of email text and labels( 0 for not spam, and 1 for spam)
emails = [
    "Get rich Quick! Click here to win a million dollars",
    "Hello, could you please review this document for me", 
    "Discounts on luxury watches and handbags",
     " meeting scheduled for tomorrw, please confirm your atendance",
     " Congratulations, You've won a free vacations ticket",
     "Project update for Monday's meeting" ,
     "Win a luxury cruise to the Bahamas!",
    "Lunch at 12:30 today?",
    "URGENT: Your account has been compromised!",
    "Notes from the design sync",
    "You won a $1000 Walmart Gift Card!!",
    "Your flight confirmation: BA293",
    "Get rich quick with this one simple trick",
    "Re: Questions about the invoice",
    "LOSE 20 LBS IN 5 DAYS - GUARANTEED",
    "Feedback on the draft proposal",
    "Claim your inheritance now",
    "Happy Birthday, Sarah!",
    "RE: Final notice regarding your debt",
    "Weekly team newsletter",
    "Meet hot singles in your area tonight",
    "Zoom link for the interview",
    "You have a pending Bitcoin transfer",
    "Can you review this code?",
    "Best prices on luxury watches online",
    "Weekend hiking plans",
    "Enlargement pills - buy 1 get 1 free",
    "Your order has been shipped",
    "WORK FROM HOME: Make $500/hour",
    "Upcoming dental appointment reminder",
    "Important security alert from 'Bank of America'",
    "New comment on your document",
    "CONGRATULATIONS! You are our 1,000,000th visitor",
    "Instructions for the office move",
    "Free trial: The secret to eternal youth",
    "Quarterly budget report attached",
    "Your PayPal account is suspended - click here",
    "Regarding our conversation earlier",
    "Act now! 90% off all designer handbags",
    "Photos from the company picnic",
    "Low interest rate mortgage for you",
    "Change in holiday hours",
    "Double your investment in 24 hours",
    "RSVP: Dave's Retirement Party",
    "You've been selected for a special prize",
    "Request for a 1-on-1 meeting",
    "Increase your website traffic instantly",
    "Welcome to the team!",
    "Pharmacy deals: Save big on prescriptions",
    "Grocery list for tonight",
    "Notice of legal action - open immediately",
    "Update to our privacy policy",
    "Unlock hidden features of your phone",
    "Can we reschedule for Tuesday?",
    "Get a high school diploma in 2 weeks",
    "Your subscription renewal notice",
    "Cheap airfare deals you won't believe",
    "Follow up on the marketing lead",
    "Verified: Your new job offer awaits",
    "Training session: Cybersecurity 101",
    "Why pay more? Get the same results for less",
    "Gym membership confirmation",
    "RE: Your recent lottery winning",
    "Missing keys in the breakroom",
    "Stop hair loss now with this serum",
    "Feedback on the UI/UX workshop",
    "Exclusive invitation: Join the elite",
    "The book you requested is available",
    "Your computer is infected! Scan now",
    "Quick question about the API",
    "Refinance your home for $0 down",
    "Neighborhood watch meeting minutes",
    "Secret admirer sent you a message",
    "Internship application status",
    "Click here for free movie downloads",
    "Maintenance scheduled for Saturday",
    "Don't wait - sale ends in 5 minutes!",
    "Re: Wedding invitation details",
    "Guaranteed ways to improve your credit score",
    "Your Amazon delivery is arriving today",
    "Mystery shopper wanted: Earn $300 today",
    "Changes to the benefits package",
    "RE: Your insurance claim status",
    "Thinking of you - hope you're well",
    "Give your bank account a boost",
    "Draft for the client presentation",
    "Top quality replica bags and shoes",
    "Password reset for your account",
    "Debt relief program - sign up today",
    "Call notes: March 24th",
    "Earn rewards by watching videos",
    "Did you leave your umbrella?",
    "Your social security number is at risk",
    "Shared folder: Project Alpha",
    "Special offer just for our VIP members",
    "Your library books are due",
    "Incredible discovery for your health",
    "New post in General Discussion",
    "New way to trade stocks without risk",
    "Thoughts on the new logo?",
    "Limited time: Claim your free sample",
    "Coffee catch-up next week?",
    "You are pre-approved for a $50k loan",
    "Your tax return has been filed",
    "See who's been looking at your profile"
]

labels = [1,0,1,0,1,0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
          , 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
          , 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1, 0
          ]

# covert text data into numerical features using count vectoriization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# split the data into training and testing sets
min_length = min(len(emails), len(labels))
emails = list(emails)[:min_length]
labels = labels[:min_length]
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2 )

#create a multinomial Naive Bayes Classifier
model = MultinomialNB()

#train the model on training data 
model.fit(x_train, y_train)
#make predicition on test data 
y_pred = model.predict(x_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report: \n", report)

#predict whether a new mail is spam or not 
new_email = ["Congratulations! you have won a lottery to Hawai"]

new_email_vectorized = vectorizer.transform(new_email)
predicted_label = model.predict(new_email_vectorized)

if predicted_label[0] == 0:
    print("Predicted as not spam.")
else:
    print("Predicted as spam.")
