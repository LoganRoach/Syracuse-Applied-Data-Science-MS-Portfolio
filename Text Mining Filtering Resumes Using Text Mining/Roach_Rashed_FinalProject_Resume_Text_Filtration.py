import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
import string
import re
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



#nltk.download('all')

# Directory Adjustment
new_path = 'C:\\Users\\logan\\OneDrive\\Documents\\Syr\\IST 736 Text Mining\\Resume Project'
os.chdir(new_path)
print("Current Working Directory: ", os.getcwd())

#Data Reading, Cleaning and Preprocessing 

resumedf = pd.read_csv('Resume.csv')

print(resumedf.head(10))

print(resumedf['Category'].value_counts())

#Pie Chart Vis for category 
plt.figure(figsize=(15,10))
resumedf['Category'].value_counts().plot(kind='pie',autopct='%1.1f%%',colors=plt.cm.coolwarm(np.linspace(0,1,3)))
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Preprocessing text 

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}
    stop_words = set(stopwords.words("english"))
    for sent in sentences:
        if any(criteria in sent for criteria in ['skills', 'education']):
            words = word_tokenize(sent)
            words = [word for word in words if word not in stop_words]
            tagged_words = pos_tag(words)
            filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
            features['feature'] += " ".join(filtered_words)
    return features

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

def process_resume_data(df):
    id = df['ID']
    category = df['Category']
    text = extract_text_from_pdf(f"data/data/{category}/{id}.pdf")
    features = preprocess_text(text)
    df['Feature'] = features['feature']
    return df

def preprocess_text2(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    processed_text = ' '.join([word for word in sentences if word not in stop_words])
    return processed_text

resumedf = resumedf.drop(["Resume_html"], axis=1)
resumedf = resumedf.apply(process_resume_data, axis=1)
resumedf = resumedf.drop(columns=['Resume_str'])
resumedf.to_csv("resume_data.csv", index=False)

resumedf = pd.read_csv("resume_data.csv")
resumedf2 = pd.read_csv('SecondSmallerDataSet.csv')
print(resumedf2.columns)
print(resumedf.head(10))
print(resumedf2.head(10))

resumedf2.rename(columns={'Resume': 'Feature'}, inplace=True)

resumedf2['Feature'] = resumedf2['Feature'].apply(preprocess_text2)

resumedf = pd.concat([resumedf, resumedf2])
print(resumedf.head(10))
print(str(resumedf))
print(resumedf.head(10))
print(str(resumedf2))

#Wordcloud by categories


resumedf['Category'] = resumedf['Category'].str.lower()
resumedf['Category'] = resumedf['Category'].astype(str)
categories = np.sort(resumedf['Category'].unique())
categories

resume_categories = [resumedf[resumedf['Category'] == category].loc[:, ['Feature', 'Category']] for category in categories]

def wordcloud(resumedf):
    txt = ' '.join(txt for txt in resumedf['Feature'])
    wordcloud = WordCloud(
        height=2000,
        width=4000
    ).generate(txt)

    return wordcloud

plt.figure(figsize=(32, 20))

for i, category in enumerate(categories):
    wc = wordcloud(resume_categories[i])

    plt.subplot(5, 5, i + 1).set_title(category)
    plt.imshow(wc)
    plt.axis('off')
    plt.plot()

plt.show()
plt.close()

#K-nearest neighbors

tdif = TfidfVectorizer(stop_words='english')
knn = OneVsRestClassifier(KNeighborsClassifier())




rdf1 = pd.read_csv('Resume.csv')
rdf2 = pd.read_csv('SecondSmallerDataSet.csv')
rdf1 = rdf1[[ 'Category', 'Resume_str']]
rdf1.rename(columns = {'Resume_str':'Resume'}, inplace = True)
RDF = pd.concat([rdf1, rdf2])
print(str(RDF))
tdif.fit(RDF['Resume'])
res_vector = tdif.fit_transform(RDF['Resume'])

X = res_vector
RDF['Category'] = RDF['Category'].astype(str)
y = RDF['Category']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# LDA Work Below


#tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#tfidf_matrix = tfidf_vectorizer.fit_transform(RDF['Resume'])

num_topics_list = [10, 100, 500, 1000]

# Iterate over each number of topics
for num_topics in num_topics_list:
    print(f"Testing with {num_topics} topics:")

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(RDF['Resume'])

    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(tfidf_matrix)

    topic_distribution = lda_model.transform(tfidf_matrix)

    X = topic_distribution
    y = RDF['Category']

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier_svc = SVC(kernel='linear')
    classifier_svc.fit(X_train, y_train)

    y_pred_svc = classifier_svc.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    print(f"SVC Accuracy with {num_topics} topics:", accuracy_svc)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred_rf = rf_classifier.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy with {num_topics} topics:", accuracy_rf)
    
# Print the top words in each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
num_top_words = 10
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic {topic_idx + 1}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

import numpy as np

# Get feature importances from the trained Random Forest model
feature_importances = rf_classifier.feature_importances_

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Get the top N most important features (topics)
top_n = 10  # Number of top features to visualize
top_features = sorted_indices[:top_n]
top_feature_importances = feature_importances[top_features]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(top_n), top_feature_importances, align='center')
plt.xticks(range(top_n), [f"Topic {i}" for i in top_features])
plt.xlabel('Feature (Topic)')
plt.ylabel('Feature Importance')
plt.title('Top 10 Most Important Features (Topics) in Random Forest Model')
plt.show()


# Get feature importances from the trained Random Forest model
feature_importances = rf_classifier.feature_importances_

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), [f"Topic {i}" for i in sorted_indices])
plt.xlabel('Feature (Topic)')
plt.ylabel('Feature Importance')
plt.title('Feature Importances in Random Forest Model')
plt.show()


def visualize_topic_terms(lda_model, feature_names, n_words=10):
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
        print()
        
def visualize_document_topics(topic_distribution):
    plt.figure(figsize=(10, 6))
    for i in range(topic_distribution.shape[1]):
        plt.hist(topic_distribution[:, i], bins=30, alpha=0.5, label=f"Topic {i}")
    plt.xlabel("Topic Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Document-Topic Distribution")
    plt.show()        
 
def visualize_word_clouds(lda_model, feature_names, n_words=50):
    for topic_idx, topic in enumerate(lda_model.components_):
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic_idx} - Word Cloud")
        plt.show()

visualize_topic_terms(lda_model, feature_names)
visualize_document_topics(topic_distribution)
visualize_word_clouds(lda_model, feature_names)        

# SVC USING NO LDA 
tfidf_vectorizer_2 = TfidfVectorizer(stop_words='english')
tfidf_matrix_2 = tfidf_vectorizer_2.fit_transform(RDF['Resume'])

X3 = tfidf_matrix_2
y3 = RDF['Category']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X3, y3, test_size=0.2, random_state=42)


classifier_SVC = SVC(kernel='linear')
classifier_SVC.fit(X_train2, y_train2)

y_pred_SVC = classifier_SVC.predict(X_test2)
accuracy_SVC = accuracy_score(y_test2, y_pred_SVC)
print("Accuracy:", accuracy_SVC)


# RF USING NO LDA
rf_classifier2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier2.fit(X_train2, y_train2)

# Evaluate Model
y_pred_RF2 = rf_classifier2.predict(X_test2)
accuracy_RF2 = accuracy_score(y_test2, y_pred_RF2)
print("Random Forest Accuracy:", accuracy_RF2)
