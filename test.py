from senticnet.senticnet import SenticNet
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

sn = SenticNet()
error_counter = 0
#print("concept_info =",sn.concept('love'))
for word in stop_words:
	try:
		
		print("label =",  sn.polarity_value(word), "score = ", sn.polarity_intense(word))

	except:
		error_counter+=1
		continue

print(error_counter,len(stop_words))



polarity_intense = sn.polarity_intense('great')
moodtags = sn.moodtags('love')
semantics = sn.semantics('love')
sentics = sn.sentics('love')