import textrazor
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk assets
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

# api call
textrazor.api_key = "f4a91b1a7141528bf31af8b9d0220448d21bcdeb9b2f6fec98006039"
# extractors for Text Razor API
client = textrazor.TextRazor(extractors=["words", "entities", "entailments",
                                         "relations", "noun_phrases", "topics"])

# product reviews from Amazon
review1 = "This camera works great and my children love it. The picture quality is okay. It works best when the subject is close. It doesn't work well for far away shots. The pictures are smaller then I would like; however, they are okay. The one drawback is the price of the film; especially, since it doesn't take long to take 10 pictures. Then a new pack needs to be put in."
review2 = "Don't waste your money. I would rate it a 0 if there was one. The picture quality is horrible. Focus is bad, flash is too bright, I went through 2 packs of film and every shot was overexposed and had a strange green tint, blurry, and off centered. A terrible camera and you will burn through a pile of film hoping you can get a good shot. Only good thing to say is that it is easy to load."
review3 = "I bought this camera for me and my husband to have quick mini photos placed in The Adventure Challenge book I also bought for our anniversary.I love the pretty sky blue color of this camera which is just so visually pleasing to look at and how easy it is to turn the camera on just by pressing the button near the lens barrel. The manual provided also has easy to read instructions with pictures.The picture quality is pretty good considering that the focus range is from 30-50 cm. I didn’t expect much for a small unprofessional camera. So, overall I’m pretty impressed with the photos.The only thing I didn’t like was that when I first tried using the camera, the charging flash would constantly blink for error and wouldn’t take a photo. I tried taking out the batteries and replacing them and “pushing them in all the way.” It took me googling and reading a Reddit post to find out that the Airdeer AA batteries that are included with this camera weren’t compatible and when I switched to Duracell brand, the camera worked instantly. Alkaline batteries are a must in order for this camera to work. This situation is the only reason I’m leaving 4 stars."

# specifying which review to analyze
response = client.analyze(review3)

# This list of words helps retrieve parts of the reviews that point to
# potential areas for improvement relating to the product
improvement_words = ["successful", "success", "better", "worse", "hope",
                     "next", "previous", "poor", "quality", "work", "expected",
                     "supposed", "newer", "older", "unsuccessful", "horrible",
                     "bad", "awful", "blurry", "strange", "cannot", "too",
                     "bright", "extremely", "centered", "pretty", "well", "not",
                     "wasted", "pictures", "picture"]


# categories
# this calls the Textrazor API to help classify the topics in the review
client.set_classifiers(["textrazor_iab_content_taxonomy"])
print("The related metadata topics for this review are:")
for topic in response.topics():
    if topic.score > 0.95:
        print(topic.label)

# lists to contain the "areas for improvement"
improvements = []
words_improvement = []

# improvements code
# This matches properties to words in the improvement_words list
# and retrieves and appends the predicate and property words to
# reconstruct the sentence that relates to the "areas for improvement"
for property in response.properties():
    for word in property.predicate_words:
        if word.lemma in improvement_words:
            words_improvement.append(property.predicate_words)
            improvements.append(property.property_words)
            break

# code to extract the predicate and property words from the TextRazor object
# that is retrieved
test = str(words_improvement[0])
test = str(re.findall('"([^"]*)"', test))

new_string1 = test.replace("b'", '')
new_string = new_string1.replace("'", '')

test1 = str(improvements[0])
test1 = str(re.findall('"([^"]*)"', test1))

new_string2 = test1.replace("b'", '')
new_string1 = new_string2.replace("'", '')

# combining the predicate and property words to reconstruct the
# "areas of improvement" sentence
goal = (new_string + new_string1)
print("An identified area for improvement relates to:")
print(goal)
# sentiment analysis
# implementation of NLTK sentiment analyzer
sentiment = SentimentIntensityAnalyzer()
print("The sentiment for the review is: ")
print(sentiment.polarity_scores(review3))



