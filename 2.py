import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import base64
from wordcloud import WordCloud, STOPWORDS
from nltk.util import ngrams
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')  # Download the pre-trained sentiment analyzer

sid = SentimentIntensityAnalyzer()
def get_sentiment_scores(df):
    sid = SentimentIntensityAnalyzer()

    # Apply polarity_scores method to 'Reviews' column
    df['sentiment'] = df['Reviews'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Define threshold for positive and negative sentiment
    threshold = 0.05

    # Count number of positive and negative reviews
    num_pos = sum(df['sentiment'] > threshold)
    num_neg = sum(df['sentiment'] < -threshold)

    # Print overall sentiment of dataframe
    if num_pos > num_neg:
        st.success('The Product review has a positive sentiment.')
    elif num_pos < num_neg:
        st.success('The Product review  has a negative sentiment.')
    else:
        st.success('The Product review  has a neutral sentiment.')
def preprocess_cleaning(text, pos_tags=None, lower=False):
    import re, string
    if isinstance(text, str):
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor'} | set()
        text = text.lower() if lower else text
        text_without_punc = text.translate(str.maketrans('', '', string.punctuation))
        text_tokens = nltk.RegexpTokenizer(r'\w+|\$[\d\.]+|\S+').tokenize(text_without_punc)
        tokens_without_sw = [word for word in text_tokens if word not in stop_words]
        tokens_lemmatized = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in tokens_without_sw]
        if pos_tags is not None:
            tokens_pos_tagged = nltk.pos_tag(tokens_lemmatized)
            tokens_lemmatized = [word for word, pos_tag in tokens_pos_tagged if
                                 pos_tag in pos_tags and word not in ['top']]  # ['JJ','JJR','JJS']
        return " ".join(tokens_lemmatized)
    else:
        return " "



# List of product review URLs
st.markdown(
    f"""
    <div style='text-align: center;'>
        <h1>Real Time Amazon Product Review Sentiment Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)
container = st.container()
url="https://logos-world.net/wp-content/uploads/2020/04/Amazon-Logo.png"
container.markdown(f'<div style="text-align: center;">'
                       f'<img src="{url}"width="150">'
                       f'</div>',
                       unsafe_allow_html=True)
Categories=['Electronics','Clothing and Fashion','Home and Kitchen','Books']
selected_category= st.selectbox("Select a product review URL", Categories)
if (selected_category=='Electronics'):
  product = ["Apple iPhone 12 Mini (128GB) - Blue","EchoDot (3rdGen) - Smartsp", "Sonya 7 III Full - FrameM irrorl", "Bose Quiet comfort 35Ii Noise", "boAt Wave Call Smart Watch", "ZEBRONICS Zeb - Dash Plus2", "Spigen Tough Armor BackCove"]
  Product_id=["B08L5VN68Y","B07PFFMP9P","B07B43WPVK","B0756CYWWD","B0B5B6PQCT","B08YDFX7Y1","B096HG9474"]

  #urls = [ "https://www.amazon.in/New-Apple-iPhone-Mini-128GB/product-reviews/B08L5VN68Y/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07PFFMP9P/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07B43WPVK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0756CYWWD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BPPWX8F2 /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0B5B6PQCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B08YDFX7Y1/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B096HG9474/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BRQV14YD /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"]
  selected_product = st.selectbox("Select a product review URL", product)
  x = product.index(selected_product)
  y = Product_id[x]
  selected_url = f"https://www.amazon.in/product-reviews/{y}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
  link = f"[For more details ..... \n You can refer this link ](https://www.amazon.in/dp/{y}/?_encoding=UTF8&pd_rd_w=21WMM&content-id=amzn1.sym.82b4a24f-081c-4d15-959c-ef13a1d3fa4e&pf_rd_p=82b4a24f-081c-4d15-959c-ef13a1d3fa4e&pf_rd_r=2FEEVAEMP0E15J8PD1ZE&pd_rd_wg=LduuX&pd_rd_r=1e90e15d-cb65-4e84-b14a-2e6216025c7e&ref_=pd_gw_ci_mcx_mr_hp_atf_m&th=1)"
  st.markdown(link, unsafe_allow_html=True)
# Create a dropdown menu to select the URL

if (selected_category=='Clothing and Fashion'):
  product=["Amazon Brand - Symbol Men","Scott International Polo T-Shir","Allen Solly Men Jet Black Re","Zeel Clothing Women's Organ","Vardha Women's Kanchipuram Raw Silk Saree with Unstitched Blouse Piece - Zari Woven Work Sarees for Wedding","KARAGIRI Women's Woven Silk Blend Saree With Blouse Piece (KARAGIRI_Purple)"]
  Product_id =["B07BDY51R6","B01F0XDZ80","B06Y2FG6R7","B09MBY3SNR","B08WCF58GP","B0B5HL5WZR"]
  #urls = [ "https://www.amazon.in/product-reviews/B07BDY51R6/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B01F0XDZ80/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B06Y2FG6R7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0756CYWWD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B09MBY3SNR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0B5B6PQCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BQDS2M4X/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B096HG9474/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BWRWQ3XK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"]
  selected_product = st.selectbox("Select a product Name", product)
  x = product.index(selected_product)
  y = Product_id[x]
  selected_url = f"https://www.amazon.in/product-reviews/{y}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
# Create a dropdown menu to select the URL
if (selected_category=='Home and Kitchen'):
  product=["Havells Ambrose 1200mm Ceiling Fan","Havells OFR - 11Fin 2900-Watt PTC Fan Heater (Black)","Havells Capture 500 Watt Mixer Grinder with 3 Bigger size Stainless Steel Jar (Grey & Green) with 5 year Motor warranty","Havells Air Fryer Prolife Digi with 4L Capacity | Digital Touch Panel | Auto On/Off | 60 Min Timer | Basket Release Button | Air Filtration System | 2 Yr Warranty, Black","Priti - Golden Coffee Table Golden with Black Laminated Marble Table top Round Metal Table and Outdoor or Indoor","Accent for Living Room Bedroom Balcony and Office - Set of Two","Pigeon-Amaze-Plus-1-5-Ltr","beatXP-Multipurpose-Portable-Electronic-Weighing","Lifelong-LLMG23-500-Watt-Liquidizing-Stainless","Philips-GC1905-1440-Watt-Steam-Spray","Kuber-Industries-Laundry-Organiser-LMESH02","NutriPro-Bullet-Juicer-Grinder-Blades","Durable-Quality-Bicycle-Ultra-Loud-Trending","Lista-Lista056-Rechargeable-Light-Bright","Quality-Bicycle-Silicone-Cycling-Cushion","Boldfit-Typhoon-Shaker-Leakproof-Preworkout","ElectroSky-Building-Exercise-Workout-Training","Bajaj-Torque-New-Honeycomb-Technology","Lifelong-LLMG23-500-Watt-Liquidizing-Stainless","Pigeon-Kessel-1-2-Litre-Multi-purpose-Kettle"]
  #urls = [ "https://www.amazon.in/New-Apple-iPhone-Mini-128GB/product-reviews/B08L5VN68Y/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07PFFMP9P/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07B43WPVK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0756CYWWD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BPPWX8F2 /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0B5B6PQCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B08YDFX7Y1/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B096HG9474/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BRQV14YD /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"]
  Product_id=["B01LYU3BZF","B00PQCXVQM","B0872HNF94","B01LWUEDJI","B0B4DPGGG9","B07WMS7TWB","B0B61DSF17","B09X5C9VLK","B008QTK47Q","B077K1HFZ3","B09J2SCVQT","B09RPRB634","B075GYJSPT","B075GYJSPT","B00RV3CLVU","B08JTMZRJW","B09YDLBS5D","B09R3QNGW5","B09X5C9VLK","B01I1LDZGA"]
# Create a dropdown menu to select the URL
  selected_product = st.selectbox("Select a product Name", product)
  x=product.index(selected_product)
  y=Product_id[x]
  selected_url=f"https://www.amazon.in/product-reviews/{y}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
if (selected_category=='Books'):
  product=["I Wish My Kids Had Cancer: A Family Surviving the Autism Epidemic ","Dopamine-Detox-Remove-Distractions-Productivity-ebook","Sanskrit-Hindi-Kosh-Vaman-Shivram-Apte","Psychology-Money-Morgan-Housel","Atomic-Habits-James-Clear","Ikigai-H%C3%A9ctor-Garc%C3%ADa","Habits-Highly-Effective-People-Powerful-ebook","Rich-Dad-Poor-Middle-Anniversary","Lifes-Amazing-Secrets-Balance-Purpose","Dont-Believe-Everything-You-Think","Mans-Search-Meaning-Viktor-Frankl","As-Man-Thinketh-James-Allen","How-Win-Friends-Influence-People","BlackBook-English-Vocabulary-March-Nikhil","NTA-Paper-Teaching-Research-Aptitude","Services-Prelims-Topic-wise-Solved-Papers","Indian-English-Revised-Services-Administrative"]
  Product_id=["1606720708","B098MHBF23","8120820975","9390166268","1847941834","178633089X","B01069X4H0","1612681131","143442295","935543135X","1846041244","9386538172","8194790891","8195645712","935606590X","B09XMHRQ63","9354600352"]
  #urls = [ "https://www.amazon.in/New-Apple-iPhone-Mini-128GB/product-reviews/B08L5VN68Y/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07PFFMP9P/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",    "https://www.amazon.in/product-reviews/B07B43WPVK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0756CYWWD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BPPWX8F2 /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0B5B6PQCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B08YDFX7Y1/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B096HG9474/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews","https://www.amazon.in/product-reviews/B0BRQV14YD /ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"]
  selected_product = st.selectbox("Select a product Name", product)
  x = product.index(selected_product)
  y = Product_id[x]
  selected_url = f"https://www.amazon.in/product-reviews/{y}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

code = requests.get(selected_url)
if code.status_code == 200:
    soup = BeautifulSoup(code.content, 'html.parser')
    names = soup.select('span.a-profile-name')[2:]
    image_tags = soup.select('img')
    image_urls = [img['src'] for img in image_tags]
    container = st.container()
    container.markdown(f'<div style="text-align: center;">'
                       f'<img src="{image_urls[3]}"width="250">'
                       f'</div>',
                       unsafe_allow_html=True)
    st.write("\n\n")
    if(len(names)>0):
        titles = soup.select('a.review-title span')
        stars = soup.select('i.review-rating span.a-icon-alt')[2:]
        reviews = soup.select('span.review-text-content span')

        cust_name = []
        ratings = []
        rev_title = []
        rev_content = []

        st.write("\n\n\n")
        minx = min(len(names), len(stars), len(titles), len(reviews))
        for i in range(minx):
            cust_name.append(names[i].get_text())
            ratings.append(stars[i].get_text())
            rev_title.append(titles[i].get_text())
            rev_content.append(reviews[i].get_text().strip("\n "))
        df = pd.DataFrame()
        df['Customer Name'] = cust_name
        df['Ratings'] = ratings
        df['Review Title'] = rev_title
        df['Reviews'] = rev_content
        df = df.dropna()
        st.write(df)

        df.to_csv(f'Amazon_Product_Reviews/{y}.csv')
        if st.button("Download CSV File"):
            csv = df.to_csv(index=False)
            print(csv)
            if csv is not None:
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{y}.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

        df = pd.read_csv(f"Amazon_Product_Reviews\{y}.csv")
        df['text_clean'] = df['Reviews'].apply(preprocess_cleaning)
        comment_words = ''
        stopwords = set(STOPWORDS)
        for val in df['text_clean']:
            val = str(val)

            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            comment_words += " ".join(tokens) + " "

        wordcloud = WordCloud(width=1000, height=800,
                              background_color='Black',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

        # Display the image
        st.image(wordcloud.to_array())

        get_sentiment_scores(df)

        # Classify the document as positive, negative, or neutral based on the average sentiment score
    else:
        st.write("\n\n")
        st.success("The Product Review has Negative Sentiment")


else:
    st.write("Error: The selected URL returned a status code of", code.status_code)
